from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.initializers import glorot_uniform as xavier_initializer
from tensorflow.keras.optimizers import RMSprop
from scipy.io import wavfile
from generator import *
from discriminator import *
import numpy as np
from data_loader import read_and_decode, de_emph
from bnorm import VBN
from ops import *
import timeit
import os

# Disable eager execution
tf.compat.v1.disable_eager_execution()

def parse_function(proto, canvas_size):
    features = {
        'wav': tf.io.FixedLenFeature([], tf.string),
        'noisy': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(proto, features)
    wav = tf.io.decode_raw(parsed_features['wav'], tf.float32)
    noisy = tf.io.decode_raw(parsed_features['noisy'], tf.float32)
    wav = tf.reshape(wav, [canvas_size])
    noisy = tf.reshape(noisy, [canvas_size])
    return wav, noisy

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
        # type of deconv
        self.deconv_type = args.deconv_type
        # specify if use biases or not
        self.bias_downconv = args.bias_downconv
        self.bias_deconv = args.bias_deconv
        self.bias_D_conv = args.bias_D_conv
        # clip D values
        self.d_clip_weights = False
        # apply VBN or regular BN?
        self.disable_vbn = False
        self.save_path = args.save_path
        # num of updates to be applied to D before G
        # this is k in original GAN paper (https://arxiv.org/abs/1406.2661)
        self.disc_updates = 1
        # set preemph factor
        self.preemph = args.preemph
        if self.preemph > 0:
            print('*** Applying pre-emphasis of {} ***'.format(self.preemph))
        else:
            print('--- No pre-emphasis applied ---')
        # canvas size
        self.canvas_size = args.canvas_size
        self.deactivated_noise = False
        # dilation factors per layer (only in atrous conv G config)
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps for AutoEncoder SEGAN (v1)
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        # Define D fmaps
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.init_noise_std = args.init_noise_std
        self.disc_noise_std = tf.Variable(self.init_noise_std, trainable=False)
        self.disc_noise_std_summ = scalar_summary('disc_noise_std',
                                                  self.disc_noise_std)
        self.e2e_dataset = args.e2e_dataset
        # G's supervised loss weight
        self.l1_weight = args.init_l1_weight
        self.l1_lambda = tf.Variable(self.l1_weight, trainable=False)
        self.deactivated_l1 = False
        # define the functions
        self.discriminator = discriminator
        # register G non linearity
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
        d_opt = RMSprop(learning_rate=config.d_learning_rate)
        g_opt = RMSprop(learning_rate=config.g_learning_rate)

        for idx, device in enumerate(self.devices):
            with tf.device("/%s" % device):
                with tf.name_scope("device_%s" % idx):
                    with variables_on_gpu0():
                        self.build_model_single_gpu(idx)
                        d_grads = d_opt.get_gradients(self.d_losses[-1], self.d_vars)
                        g_grads = g_opt.get_gradients(self.g_losses[-1], self.g_vars)
                        all_d_grads.append(d_grads)
                        all_g_grads.append(g_grads)
                        tf.compat.v1.get_variable_scope().reuse_variables()
        avg_d_grads = average_gradients(all_d_grads)
        avg_g_grads = average_gradients(all_g_grads)
        self.d_opt = d_opt.apply_gradients(zip(avg_d_grads, self.d_vars))
        self.g_opt = g_opt.apply_gradients(zip(avg_g_grads, self.g_vars))

    def build_model_single_gpu(self, gpu_idx):
        if gpu_idx == 0:
            # Create a dataset from the TFRecord file
            dataset = tf.data.TFRecordDataset([self.e2e_dataset])
            # Parse the TFRecord
            dataset = dataset.map(lambda x: parse_function(x, self.canvas_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Shuffle, repeat, and batch the examples
            dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).repeat()
            # Create an iterator
            iterator = iter(dataset)
            self.get_wav, self.get_noisy = iterator.get_next()

        if gpu_idx == 0:
            self.Gs = []
            self.zs = []
            self.gtruth_wavs = []
            self.gtruth_noisy = []

        self.gtruth_wavs.append(self.get_wav)
        self.gtruth_noisy.append(self.get_noisy)

        # add channels dimension to manipulate in D and G
        wavbatch = tf.expand_dims(self.get_wav, -1)
        noisybatch = tf.expand_dims(self.get_noisy, -1)
        # by default leaky relu is used
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
                    self.alpha_summ.append(histogram_summary('alpha_{}'.format(m), ref_alpha))
            dummy_joint = tf.concat([wavbatch, noisybatch], axis=2)
            dummy = self.discriminator(self, dummy_joint, reuse=False)

        G, z = self.generator(noisybatch, is_ref=False, spk=None, do_prelu=do_prelu)
        self.Gs.append(G)
        self.zs.append(z)

        D_rl_joint = tf.concat([wavbatch, noisybatch], axis=2)
        D_fk_joint = tf.concat([G, noisybatch], axis=2)
        d_rl
