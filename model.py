from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.initializers import glorot_uniform as xavier_initializer
from tensorflow.keras.optimizers import RMSprop
from scipy.io import wavfile
from generator import AEGenerator, Generator
from discriminator import discriminator
import numpy as np
from data_loader import read_and_decode, de_emph
from bnorm import VBN
from ops import *
import timeit
import os


class Model(object):

    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.compat.v1.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name),
                        global_step=step)

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.compat.v1.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True


class SEGAN(Model):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, sess, args, devices, infer=False, name='SEGAN'):
        super(SEGAN, self).__init__(name)
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.d_label_smooth = args.d_label_smooth
        self.devices = devices
        self.z_dim = args.z_dim
        self.z_depth = args.z_depth
        self.deconv_type = args.deconv_type
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        self.d_clip_weights = False
        self.disable_vbn = False
        self.save_path = args.save_path
        self.disc_updates = 1
        self.preemph = args.preemph
        if self.preemph > 0:
            print('*** Applying pre-emphasis of {} ***'.format(self.preemph))
        else:
            print('--- No pre-emphasis applied ---')
        self.canvas_size = args.canvas_size
        self.deactivated_noise = False
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.disc_noise_std_summ = scalar_summary('disc_noise_std',
                                                  self.disc_noise_std)
        self.e2e_dataset = args.e2e_dataset
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        self.discriminator = discriminator
        self.g_nl = args.g_nl
        if args.g_type == 'ae':
            self.generator = AEGenerator(self)
        elif args.g_type == 'dwave':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(args.g_type))
        self.build_model(args)

    def build_model(self, config):
        all_d_grads = []
        all_g_grads = []
        d_opt = tf.compat.v1.train.RMSPropOptimizer(config.d_learning_rate)
        g_opt = tf.compat.v1.train.RMSPropOptimizer(config.g_learning_rate)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(idx)
                        d_grads = d_opt.compute_gradients(self.d_losses[-1],
                                                          var_list=self.d_vars)
                        g_grads = g_opt.compute_gradients(self.g_losses[-1],
                                                          var_list=self.g_vars)
                        all_d_grads.append(d_grads)
                        all_g_grads.append(g_grads)
                        tf.compat.v1.get_variable_scope().reuse_variables()
        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        self.d_opt = d_opt.apply_gradients(avg_d_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            dataset = tf.data.TFRecordDataset([self.e2e_dataset])
            dataset = dataset.map(lambda x: parse_function(x, self.canvas_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat()
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            self.get_wav, self.get_noisy = iterator.get_next()

        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []

        self.gtruth_wavs.append(self.get_wav)
        self.gtruth_noisy.append(self.get_noisy)

        wavbatch = tf.expand_dims(self.get_wav, -1)
        noisybatch = tf.expand_dims(self.get_noisy, -1)
        do_prelu = False
        if self.g_nl == 'prelu':
            do_prelu = True
        if gpu_idx == 0:
            ref_Gs = self.generator(noisybatch, is_ref=True,
                                    spk=None,
                                    do_prelu=do_prelu)
            print('num of G returned: ', len(ref_Gs))
            self.reference_G = ref_Gs[0]
            self.ref_z = ref_Gs[1]
            if do_prelu:
                self.ref_alpha = ref_Gs[2:]
                self.alpha_summ = []
                for m, ref_alpha in enumerate(self.ref_alpha):
                    self.alpha_summ.append(histogram_summary('alpha_{}'.format(m),
                                                             ref_alpha))
            dummy_joint = tf.concat([wavbatch, noisybatch], axis=2)
            dummy = self.discriminator(self, dummy_joint, reuse=False)

        G, z  = self.generator(noisybatch, is_ref=False, spk=None,
                               do_prelu=do_prelu)
        self.Gs.append(G)
        self.zs.append(z)

        D_rl_joint = tf.concat([wavbatch, noisybatch], axis=2)
        D_fk_joint = tf.concat([G, noisybatch], axis=2)
        d_rl_logits = self.discriminator(self, D_rl_joint, reuse=True)
        d_fk_logits = self.discriminator(self, D_fk_joint, reuse=True)

        self.d_rl_sum = histogram_summary("d_real", d_rl_logits)
        self.d_fk_sum = histogram_summary("d_fake", d_fk_logits)
        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.gen_audio_summ = audio_summary('G_audio', G)
        self.gen_summ = histogram_summary('G_wav', G)

        if gpu_idx == 0:
            self.g_losses = []
            self.g_l1_losses = []
            self.g_adv_losses = []
            self.d_rl_losses = []
            self.d_fk_losses = []
            self.d_losses = []

        d_rl_loss = tf.reduce_mean(tf.square(d_rl_logits - 1))
        d_fk_loss = tf.reduce_mean(tf.square(d_fk_logits))
        g_adv_loss = tf.reduce_mean(tf.square(d_fk_logits - 1))

        d_loss = d_rl_loss + d_fk_loss
        g_l1_loss = self.l1_lambda * tf.reduce_mean(tf.abs(G - wavbatch))

        g_loss = g_adv_loss + g_l1_loss

        self.g_l1_losses.append(g_l1_loss)
        self.g_adv_losses.append(g_adv_loss)
        self.g_losses.append(g_loss)
        self.d_rl_losses.append(d_rl_loss)
        self.d_fk_losses.append(d_fk_loss)
        self.d_losses.append(d_loss)

        self.d_rl_loss_sum = scalar_summary("d_rl_loss", d_rl_loss)
        self.d_fk_loss_sum = scalar_summary("d_fk_loss", d_fk_loss)
        self.g_loss_sum = scalar_summary("g_loss", g_loss)
        self.g_loss_l1_sum = scalar_summary("g_l1_loss", g_l1_loss)
        self.g_loss_adv_sum = scalar_summary("g_adv_loss", g_adv_loss)
        self.d_loss_sum = scalar_summary("d_loss", d_loss)

        if gpu_idx == 0:
            self.get_vars()


    def get_vars(self):
        t_vars = tf.compat.v1.trainable_variables()
        self.d_vars_dict = {}
        self.g_vars_dict = {}
        for var in t_vars:
            if var.name.startswith('d_'):
                self.d_vars_dict[var.name] = var
            if var.name.startswith('g_'):
                self.g_vars_dict[var.name] = var
        self.d_vars = list(self.d_vars_dict.values())
        self.g_vars = list(self.g_vars_dict.values())
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars
        if self.d_clip_weights:
            print('Clipping D weights')
            self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05)) for v in self.d_vars]
        else:
            print('Not clipping D weights')

    def vbn(self, tensor, name):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored):
                    self.reference_output = tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def train(self, config, devices):
        print('Initializing optimizers...')
        d_opt = self.d_opt
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.compat.v1.global_variables_initializer()
        except AttributeError:
            init = tf.compat.v1.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        g_summs = [self.d_fk_sum,
                   self.d_fk_loss_sum,
                   self.g_loss_sum,
                   self.g_loss_l1_sum,
                   self.g_loss_adv_sum,
                   self.gen_summ,
                   self.gen_audio_summ]
        if hasattr(self, 'alpha_summ'):
            g_summs += self.alpha_summ
        self.g_sum = tf.compat.v1.summary.merge(g_summs)
        self.d_sum = tf.compat.v1.summary.merge([self.d_loss_sum,
                                       self.d_rl_sum,
                                       self.d_rl_loss_sum,
                                       self.rl_audio_summ,
                                       self.real_w_summ,
                                       self.disc_noise_std_summ])

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.compat.v1.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'),
                                            self.sess.graph)

        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        sample_noisy, sample_wav, sample_z = self.sess.run([self.gtruth_noisy[0],
                                                            self.gtruth_wavs[0],
                                                            self.zs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        print('sample z shape: ', sample_z.shape)

        save_path = config.save_path
        counter = 0
        num_examples = 0
        for record in tf.compat.v1.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        num_batches = num_examples // self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        d_fk_losses = []
        d_rl_losses = []
        g_adv_losses = []
        g_l1_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:
                    for d_iter in range(self.disc_updates):
                        _d_opt, _d_sum, d_fk_loss, d_rl_loss = self.sess.run([d_opt, self.d_sum,
                                                                              self.d_fk_losses[0],
                                                                              self.d_rl_losses[0]])
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)

                    _g_opt, _g_sum, g_adv_loss, g_l1_loss = self.sess.run([g_opt, self.g_sum,
                                                                           self.g_adv_losses[0],
                                                                           self.g_l1_losses[0]])
                else:
                    for d_iter in range(self.disc_updates):
                        _d_opt, d_fk_loss, d_rl_loss = self.sess.run([d_opt,
                                                                      self.d_fk_losses[0],
                                                                      self.d_rl_losses[0]])
                        if self.d_clip_weights:
                            self.sess.run(self.d_clip)

                    _g_opt, g_adv_loss, g_l1_loss = self.sess.run([g_opt, self.g_adv_losses[0],
                                                                   self.g_l1_losses[0]])
                end = timeit.default_timer()
                batch_timings.append(end - start)
                d_fk_losses.append(d_fk_loss)
                d_rl_losses.append(d_rl_loss)
                g_adv_losses.append(g_adv_loss)
                g_l1_losses.append(g_l1_loss)
                print('{}/{} (epoch {}), d_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, g_l1_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    d_rl_loss,
                                                    d_fk_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter // num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    fdict = {self.gtruth_noisy[0]:sample_noisy,
                             self.zs[0]:sample_z}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)
                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m,
                                                           np.max(canvas_w[m]),
                                                           np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path,
                                                   'sample_{}-'
                                                   '{}.wav'.format(counter, m)),
                                      16000,
                                      de_emph(canvas_w[m],
                                              self.preemph))
                        m_gtruth_path = os.path.join(save_path, 'gtruth_{}.'
                                                                'wav'.format(m))
                        if not os.path.exists(m_gtruth_path):
                            wavfile.write(os.path.join(save_path,
                                                       'gtruth_{}.'
                                                       'wav'.format(m)),
                                          16000,
                                          de_emph(swaves[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'noisy_{}.'
                                                       'wav'.format(m)),
                                          16000,
                                          de_emph(sample_noisy[m],
                                                  self.preemph))
                            wavfile.write(os.path.join(save_path,
                                                       'dif_{}.wav'.format(m)),
                                          16000,
                                          de_emph(sample_dif[m],
                                                  self.preemph))
                        np.savetxt(os.path.join(save_path, 'd_rl_losses.txt'),
                                   d_rl_losses)
                        np.savetxt(os.path.join(save_path, 'd_fk_losses.txt'),
                                   d_fk_losses)
                        np.savetxt(os.path.join(save_path, 'g_adv_losses.txt'),
                                   g_adv_losses)
                        np.savetxt(os.path.join(save_path, 'g_l1_losses.txt'),
                                   g_l1_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    batch_idx = 0
                    if curr_epoch >= config.l1_remove_epoch and not self.deactivated_l1:
                        print('** Deactivating L1 factor! **')
                        self.sess.run(tf.compat.v1.assign(self.l1_lambda, 0.))
                        self.deactivated_l1 = True
                    if curr_epoch >= config.denoise_epoch and not self.deactivated_noise:
                        decay = config.noise_decay
                        if not hasattr(self, 'curr_noise_std'):
                            self.curr_noise_std = self.init_noise_std
                        new_noise_std = decay * self.curr_noise_std
                        if new_noise_std < config.denoise_lbound:
                            print('New noise std {} < lbound {}, setting 0.'.format(new_noise_std, config.denoise_lbound))
                            print('** De-activating noise layer **')
                            new_noise_std = 0.
                            self.deactivated_noise = True
                        else:
                            print('Applying decay {} to noise std {}: {}'.format(decay, self.curr_noise_std, new_noise_std))
                        self.sess.run(tf.compat.v1.assign(self.disc_noise_std, new_noise_std))
                        self.curr_noise_std = new_noise_std
                if curr_epoch >= config.epoch:
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    self.writer.add_summary(_d_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('[!] Reached queues limits in training loop')
        finally:
            coord.request_stop()
        coord.join(threads)

    def clean(self, x):
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i  < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = self.canvas_size - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            fdict = {self.gtruth_noisy[0]:x_}
            canvas_w = self.sess.run(self.Gs[0],
                                     feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            print('canvas w shape: ', canvas_w.shape)
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        c_res = de_emph(c_res, self.preemph)
        return c_res


class SEAE(Model):
    """ Speech Enhancement Auto Encoder """
    def __init__(self, sess, args, devices, infer=False):
        self.args = args
        self.sess = sess
        self.keep_prob = 1.
        if infer:
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        else:
            self.keep_prob = 0.5
            self.keep_prob_var = tf.Variable(self.keep_prob, trainable=False)
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.devices = devices
        self.save_path = args.save_path
        self.canvas_size = args.canvas_size
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.e2e_dataset = args.e2e_dataset
        self.generator = AEGenerator(self)
        self.build_model(args)

    def build_model(self, config):
        all_g_grads = []
        g_opt = tf.compat.v1.train.AdamOptimizer(config.g_learning_rate, config.beta_1)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(idx)
                        g_grads = g_opt.compute_gradients(self.g_losses[-1],
                                                          var_list=self.g_vars)
                        all_g_grads.append(g_grads)
                        tf.compat.v1.get_variable_scope().reuse_variables()
        avg_g_grads = average_gradients(all_g_grads)
        self.g_opt = g_opt.apply_gradients(avg_g_grads)


    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            dataset = tf.data.TFRecordDataset([self.e2e_dataset])
            dataset = dataset.map(lambda x: parse_function(x, self.canvas_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat()
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            self.get_wav, self.get_noisy = iterator.get_next()

        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []

        self.gtruth_wavs.append(self.get_wav)
        self.gtruth_noisy.append(self.get_noisy)

        wavbatch = tf.expand_dims(self.get_wav, -1)
        noisybatch = tf.expand_dims(self.get_noisy, -1)
        if gpu_idx == 0:
            self.reference_G = self.generator(noisybatch, is_ref=True,
                                              spk=None, z_on=False)

        G = self.generator(noisybatch, is_ref=False, spk=None, z_on=False)
        print('GAE shape: ', G.get_shape())
        self.Gs.append(G)

        self.rl_audio_summ = audio_summary('real_audio', wavbatch)
        self.real_w_summ = histogram_summary('real_wav', wavbatch)
        self.noisy_audio_summ = audio_summary('noisy_audio', noisybatch)
        self.noisy_w_summ = histogram_summary('noisy_wav', noisybatch)
        self.gen_audio_summ = audio_summary('G_audio', G)
        self.gen_summ = histogram_summary('G_wav', G)

        if gpu_idx == 0:
            self.g_losses = []

        g_loss = tf.reduce_mean(tf.abs(G - wavbatch))

        self.g_losses.append(g_loss)

        self.g_loss_sum = scalar_summary("g_loss", g_loss)

        if gpu_idx == 0:
            self.get_vars()

    def get_vars(self):
        t_vars = tf.compat.v1.trainable_variables()
        self.g_vars = [var for var in t_vars if var.name.startswith('g_')]
        for x in t_vars:
            assert x in self.g_vars, x.name
        self.all_vars = t_vars

    def train(self, config, devices):
        print('Initializing optimizer...')
        g_opt = self.g_opt
        num_devices = len(devices)

        try:
            init = tf.compat.v1.global_variables_initializer()
        except AttributeError:
            init = tf.compat.v1.initialize_all_variables()

        print('Initializing variables...')
        self.sess.run(init)
        self.saver = tf.compat.v1.train.Saver()
        self.g_sum = tf.compat.v1.summary.merge([self.g_loss_sum,
                                       self.gen_summ,
                                       self.rl_audio_summ,
                                       self.real_w_summ,
                                       self.gen_audio_summ])

        if not os.path.exists(os.path.join(config.save_path, 'train')):
            os.makedirs(os.path.join(config.save_path, 'train'))

        self.writer = tf.compat.v1.summary.FileWriter(os.path.join(config.save_path,
                                                         'train'),
                                            self.sess.graph)

        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        print('Sampling some wavs to store sample references...')
        sample_noisy, sample_wav = self.sess.run([self.gtruth_noisy[0],
                                                  self.gtruth_wavs[0]])
        print('sample noisy shape: ', sample_noisy.shape)
        print('sample wav shape: ', sample_wav.shape)
        save_path = config.save_path
        counter = 0
        num_examples = 0
        for record in tf.compat.v1.python_io.tf_record_iterator(self.e2e_dataset):
            num_examples += 1
        print('total examples in TFRecords {}: {}'.format(self.e2e_dataset,
                                                          num_examples))
        num_batches = num_examples // self.batch_size

        print('Batches per epoch: ', num_batches)

        if self.load(self.save_path):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        g_losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if counter % config.save_freq == 0:
                    _g_opt, _g_sum, g_loss = self.sess.run([g_opt, self.g_sum,
                                                            self.g_losses[0]])
                else:
                    _g_opt, g_loss = self.sess.run([g_opt, self.g_losses[0]])

                end = timeit.default_timer()
                batch_timings.append(end - start)
                g_losses.append(g_loss)
                print('{}/{} (epoch {}), g_loss = {:.5f},'
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    config.epoch * num_batches,
                                                    curr_epoch,
                                                    g_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))
                batch_idx += num_devices
                counter += num_devices
                if (counter // num_devices) % config.save_freq == 0:
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    fdict = {self.gtruth_noisy[0]:sample_noisy}
                    canvas_w = self.sess.run(self.Gs[0],
                                             feed_dict=fdict)
                    swaves = sample_wav
                    sample_dif = sample_wav - sample_noisy
                    for m in range(min(20, canvas_w.shape[0])):
                        print('w{} max: {} min: {}'.format(m, np.max(canvas_w[m]), np.min(canvas_w[m])))
                        wavfile.write(os.path.join(save_path, 'sample_{}-{}.wav'.format(counter, m)), 16000, canvas_w[m])
                        if not os.path.exists(os.path.join(save_path, 'gtruth_{}.wav'.format(m))):
                            wavfile.write(os.path.join(save_path, 'gtruth_{}.wav'.format(m)), 16000, swaves[m])
                            wavfile.write(os.path.join(save_path, 'noisy_{}.wav'.format(m)), 16000, sample_noisy[m])
                            wavfile.write(os.path.join(save_path, 'dif_{}.wav'.format(m)), 16000, sample_dif[m])
                        np.savetxt(os.path.join(save_path, 'g_losses.txt'), g_losses)

                if batch_idx >= num_batches:
                    curr_epoch += 1
                    batch_idx = 0
                if curr_epoch >= config.epoch:
                    print('Done training; epoch limit {} '
                          'reached.'.format(self.epoch))
                    print('Saving last model at iteration {}'.format(counter))
                    self.save(config.save_path, counter)
                    self.writer.add_summary(_g_sum, counter)
                    break
        except tf.errors.OutOfRangeError:
            print('[!] Reached queues limits in training loop')
        finally:
            coord.request_stop()
        coord.join(threads)
