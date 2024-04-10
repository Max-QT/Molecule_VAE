import six
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import pandas as pd
from keras import backend as K
import tensorflow as tf
import os
import csv
from collections import OrderedDict
from collections import Iterable


#参照keras.callback.CSVLogger
class RmseCallback(Callback):
    def __init__(self, X_test, Y_test, params, df_norm: pd.DataFrame = None, append=True, separator=','):
        super(RmseCallback, self).__init__()
        self.df_norm = df_norm
        self.X_test = X_test
        self.Y_test = Y_test
        self.config = params
        self.append = append
        self.filename = params["RMSE_file"]
        self.keys = None
        self.writer = None
        self.sep = separator
        self.append_header = True
        # self.file_flags = 'b'

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs=None):
        df_norm = self.df_norm
        X_test = self.X_test
        Y_test = self.Y_test
        y_pred = self.model.predict(X_test, self.config['batch_size'])

        if type(y_pred) is list:
            if 'reg_prop_tasks' in self.config and 'logit_prop_tasks' in self.config:
                y_pred = y_pred[-2]
            elif 'reg_prop_tasks' in self.config:
                y_pred = y_pred[-1]  # ?

        # 还原归一化
        y_pred = (y_pred - df_norm['mean'].values) / df_norm['std'].values
        Y_test = (Y_test - df_norm['mean'].values) / df_norm['std'].values

        rmse_all = np.mean(np.sqrt(np.mean(np.square(y_pred - Y_test), axis=0)), axis = 0)   # 每一个类别的误差均值
        mae_all = np.mean(np.mean(np.abs(y_pred - Y_test), axis=0), axis = 0)   # 每一个类别的误差均值

        logs = {}
        for i, tasks in enumerate(self.config["reg_prop_tasks"]):
            logs['RMSE_'+tasks] = rmse_all[i]
            logs['MAE_'+tasks] = rmse_all[i]


        if "reg_prop_tasks" in self.config:
            print("RMSE test set:{} {}".format(self.config["reg_prop_tasks"], [round(num, 4) for num in rmse_all]))
            print("MAE test set:{} {}".format(self.config["reg_prop_tasks"], [round(num, 4) for num in mae_all]))
        else:
            print("RMSE test set:", rmse_all)
            print("MAE test set:", mae_all)

        # 优点：可以写到同一个文件，缺点：每次写入都要读入整个csv文件，如果是用文件流的方式写入会更快
        # csv_file = pd.read_csv(self.filename)
        # csv_file.loc[epoch, 'RMSE_all'] = rmse_all
        # csv_file.loc[epoch, 'MAE_all'] = mae_all
        # for i, tasks in enumerate(self.config["reg_prop_tasks"]):
        #     csv_file.loc[epoch, 'RMSE'+tasks] = rmse_class[i]
        #     csv_file.loc[epoch, 'MAE'+tasks] = mae_class[i]
        # csv_file.to_csv(self.filename)

        # 参照keras.callback.CSVLogger
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join([round(i, 4) for i in k]))  # 保留四位小数
            else:
                return round(k, 4)  # 保留四位小数

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            self.keys = sorted(logs.keys())

            # class CustomDialect(csv.excel):
            #     delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=self.sep)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class WeightAnnealer_epoch(Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, schedule, weight, weight_orig, weight_name):
        super(WeightAnnealer_epoch, self).__init__()
        self.schedule = schedule
        self.weight_var = weight
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)
        new_value = new_weight * self.weight_orig
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))
        assert type(
            new_weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.weight_var, new_value)


class Weight_epoch(Callback):
    def __init__(self, weight, weight_name):
        super(Weight_epoch, self).__init__()
        self.weight_var = weight
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Current {} is {}".format(self.weight_name, sess.run(self.weight_var)))


# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num):
    return float(1)


def sigmoid_schedule(time_step, slope=1., start=None):
    return float(1 / (1. + np.exp(slope * (start - float(time_step)))))


def sample(a, temperature=0.01):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class EncoderDecoderCheckpoint(ModelCheckpoint):
    """Adapted from ModelCheckpoint, but for saving Encoder, Decoder and property
    """

    def __init__(self, encoder_model, decoder_model, params, prop_pred_model=None,
                 prop_to_monitor='val_x_pred_categorical_accuracy', save_best_only=True, monitor_op=np.greater,
                 monitor_best_init=-np.Inf):
        # Saves models at the end of every epoch if they are better than previous models
        # prop_to_montior : a property that is a valid name in the model
        # monitor_op : The operation to use when monitoring the property 
        #    (e.g. accuracy to be maximized so use np.greater, loss to minimized, so use np.loss)
        # monitor_best_init : starting point for monitor (use -np.Inf for maximization tests, and np.Inf for minimization tests)

        self.p = params
        super(ModelCheckpoint, self).__init__()
        self.save_best_only = save_best_only
        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init
        self.verbose = 1
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.prop_pred_model = prop_pred_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # self.epochs_since_last_save += 1
        # if self.epochs_since_last_save >= self.period:
        #    self.epochs_since_last_save = 0
        # filepath = self.filepath.format(epoch=epoch, **logs)
        print("--------------------------------------------------------------")
        print('val_x_pred_categorical_accuracy: ', logs.get(self.monitor))
        print("--------------------------------------------------------------")
        if not os.path.exists(self.p['checkpoint_path']):
            os.makedirs(self.p['checkpoint_path'])
        if not os.path.exists(self.p['encoder_weights_file'].split('encoder.h5')[0]):
            os.makedirs(self.p['encoder_weights_file'].split('encoder.h5')[0])

        if self.save_best_only:
            if epoch == self.p['prev_epochs']:
                file_name = os.listdir(self.p['checkpoint_path'])
                if file_name:
                    t = [os.path.getmtime(os.path.join(self.p['checkpoint_path'], file_name[i])) for i in range(len(file_name))]
                    self.best = float(file_name[t.index(max(t))].split('_')[-1].split('.h5')[0]) # 选出最晚创建的文件
                else:
                    current = logs.get(self.monitor)
                    self.best = current
                    self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}_{:.4f}.h5'.format(epoch, current)))
                    self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}_{:.4f}.h5'.format(epoch, current)))
                    if self.prop_pred_model is not None:
                        self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_{}_{:.4f}.h5'.format(epoch, current)))
            else:
                current = logs.get(self.monitor)
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model'
                              % (epoch, self.monitor, self.best, current))
                    self.best = current
                    self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}_{:.4f}.h5'.format(epoch, current)))
                    self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}_{:.4f}.h5'.format(epoch, current)))

                    # self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_best.h5'))
                    # self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_best.h5'))

                    if self.prop_pred_model is not None:
                        self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_{}_{:.4f}.h5'.format(epoch, current)))
                        # self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_best.h5'))
                    if len(os.listdir(self.p['checkpoint_path'])) > 15:
                        file_name = os.listdir(self.p['checkpoint_path'])
                        t = [os.path.getmtime(os.path.join(self.p['checkpoint_path'], file_name[i])) for i in range(len(file_name))]
                        t_1 = t.copy()
                        t.sort()
                        x = [t_1.index(t[i]) for i in range(3)]  # 选出最早创建的三个文件
                        os.remove(os.path.join(self.p['checkpoint_path'], file_name[x[0]]))
                        os.remove(os.path.join(self.p['checkpoint_path'], file_name[x[1]]))
                        os.remove(os.path.join(self.p['checkpoint_path'], file_name[x[2]]))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to ' % (epoch))
            self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}.h5'.format(epoch)))
            self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}.h5'.format(epoch)))
            if self.prop_pred_model is not None:
                self.prop_pred_model.save(os.path.join(self.p['checkpoint_path'], 'prop_pred_{}.h5'.format(epoch)))
