# -*- coding: utf-8 -*-
import torch
import numpy as np
from network import ResNet, AutoEncoder, FreezeUnet
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

def random_batch(data_cube, index, z, min, max):
    output = np.zeros(shape=(1, 1, 1024, 352), dtype='float32')
    data_img = input_data_norm(data_cube[:, index, :], min, max)
    # phase_img = z[:, index, :]
    output[0, 0, :, :] = data_img
    # output[0, 1, :, :] = phase_img
    # output[0, 1, :, :] = z
    return output
def random_batch_freeze(data_cube, index, index2):
    output = np.zeros(shape=(1, 1, 384, 352), dtype='float32')
    data_img = input_data_norm(data_cube[:, index, :])
    output[0, 0, :, :] = data_img[index2*320:index2*320+384]
    return output
def random_batch_freeze_2(data_cube, index):
    output = np.zeros(shape=(1, 1, 1024, 352), dtype='float32')
    data_img = input_data_norm(data_cube[:, index, :])
    output[0, 0, :, :] = data_img
    return output
def input_data_norm(input_cube_img, mean, deviation):
    # input_cube_img_min = np.min(input_cube_img)
    # input_cube_img_max = np.max(input_cube_img)
    input_cube_img_norm = (input_cube_img - mean) / deviation
    return input_cube_img_norm
# def random_batch_freeze(data_cube, index):
#     # shuffle(batch_list)
#     inline_list = []
#     for i in range(48):
#         inline_list.append(np.random.randint(low=0, high=1))
#     data_output = np.zeros(shape=(48, 1, 1024, 352), dtype='float32')
#     # label_output = np.zeros(shape=(48, 1024, 352), dtype='float32')
#     for i in range(48):
#         if inline_list[i] == 0:
#             data_output[i, 0, :, :] = data_cube[:, index*48+i, :]
#             # label_output[i, :, :] = label_cube[:, batch_list[i], :]
#         elif inline_list[i] == 1:
#             data_output[i, 0, :, :] = data_cube[::-1, index*48+i, :]
#             # label_output[i, :, :] = label_cube[::-1, batch_list[i], :]
#     return data_output
def model_predict_result(model_output):
    predict_result = torch.nn.Softmax(dim=1)(model_output)
    class_pos, class_no = torch.max(predict_result, 1, keepdim=True)
    model_output_index = torch.squeeze(class_no).cpu().detach().numpy()
    return model_output_index
def predict_whole_cube(data_file, model_para_file):
    seismic_data = np.load(data_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[0:1024, :, 21:]
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    torch.cuda.empty_cache()
    network = ResNet(n_classes=6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()
    model_path = model_para_file
    network.load_state_dict(torch.load(model_path))
    model_predict = np.zeros(shape=seismic_data.shape, dtype='float32')
    for i in range(seismic_data.shape[1]):
        data_input = random_batch(seismic_data, i)
        data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        model_output = network(data_input)
        model_predict[:, i, :] = model_predict_result(model_output=model_output)
        print('\rThe %d inline img!'%i)
    np.save(r"predict1.npy", model_predict)
    return
def predict_whole_cube_AE(data_file, model_para_file):
    seismic_data = np.load(data_file)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[0:1024, :392, 21:373]
    mean = np.mean(seismic_data)
    deviation = np.var(seismic_data) ** 0.5
    # seismic_data_mean = np.mean(seismic_data)
    # seismic_data_deviation = np.var(seismic_data)
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    # seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    # seismic_data = seismic_data[:512, :, :]
    torch.cuda.empty_cache()
    network = AutoEncoder(in_channels=1, out_channels=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()
    model_path = model_para_file
    network.load_state_dict(torch.load(model_path))
    model_predict = np.zeros(shape=seismic_data.shape, dtype='float32')
    # z_index = np.zeros(shape=(1024, 352), dtype='float32')
    # for i in range(352):
    #     z_index[:, i] = i * 0.01 - 1.26
    # z_index = np.load(r"Ins_phase.npy")
    for i in range(seismic_data.shape[1]):
        data_input = random_batch(seismic_data, i, 1, mean, deviation)
        data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        model_output = network(data_input)
        # model_predict[:, i, :] = model_predict_result(model_output=model_output)
        a = torch.squeeze(model_output).cpu().detach().numpy()
        model_predict[:, i, :] = a
        print('\rThe %d inline img!'%i)
    print("model_predict shape", model_predict.shape)
    np.save(r"predictAE.npy", model_predict)
    return
def predict_whole_cube_FreezeUnet(data_file, model_para_file):
    seismic_data = np.load(data_file)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = seismic_data[0:1024, :392, 21:373]
    mean = np.mean(seismic_data)
    deviation = np.var(seismic_data) ** 0.5
    # seismic_data_mean = np.mean(seismic_data)
    # seismic_data_deviation = np.var(seismic_data)
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    # seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    torch.cuda.empty_cache()
    network = FreezeUnet(in_channels=1, out_channels=6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()
    model_path = model_para_file
    network.load_state_dict(torch.load(model_path))
    model_predict = np.zeros(shape=seismic_data.shape, dtype='float32')
    # single_predict = np.zeros(shape=(1024, 352), dtype='float32')
    # for i in range(seismic_data.shape[1]):
    # z_index = np.zeros(shape=(1024, 352), dtype='float32')
    # for i in range(352):
    #     z_index[:, i] = i * 0.01 - 1.26
    # z_index = np.load(r"Ins_phase.npy")
    for i in range(seismic_data.shape[1]):
        # predict_img = []
        # single_predict = np.zeros(shape=(1024, 352), dtype='float32')
        # for j in range(3):
        #     data_input = random_batch_freeze(seismic_data, i, j)
        #     data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        #     model_output = network(data_input)
        #     predict_img.append(model_predict_result(model_output=model_output))
        # single_predict[:352, :] = predict_img[0][:352]
        # single_predict[352:672, :] = predict_img[1][32:352]
        # single_predict[672:1024, :] = predict_img[2][32:]
        # # model_predict[:, i, :] = model_predict_result(model_output=model_output)
        # # model_predict[:, i, :] = model_predict_result(model_output=model_output)
        # model_predict[:, i, :] = single_predict
        data_input = random_batch(seismic_data, i, 1, mean, deviation)
        data_input = (torch.autograd.Variable(torch.Tensor(data_input).float())).to(device)
        model_output = network(data_input)
        predict_whole = model_predict_result(model_output=model_output)
        model_predict[:, i, :] = predict_whole
        print('\rThe %d inline img!'%i, np.min(predict_whole), np.max(predict_whole))
    np.save(r"predictFreeze.npy", model_predict)
    return
if __name__ == "__main__":
    print('hello')
    # predict_whole_cube(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"saved_model99.pt")
    # predict_whole_cube_AE(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"saved_modelAE10.pt")
    predict_whole_cube_FreezeUnet(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"saved_modelFreeze10.pt")