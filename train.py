# -*- coding: utf-8 -*-
import numpy as np
from network import ResNet, AutoEncoder, FreezeUnet
import torch
from torch import nn
from collections import Counter
import pandas as pd
import segyio
import matplotlib.pyplot as plt
from random import shuffle
def random_batch(data_cube, label_cube, batch_list):
    # batch_list.pop(46, 196, 346)
    shuffle(batch_list)
    # batch_list = batch_list[:47]
    # full_img_list = [46, 196, 346]
    # index_new = np.random.randint(low=0, high=3)
    # batch_list.append(full_img_list[index_new])
    # shuffle(batch_list)
    data_output = np.zeros(shape=(32, 1, 1024, 480), dtype='float32')
    label_output = np.zeros(shape=(32, 1024, 480), dtype='float32')
    for i in range(32):
        data_output[i, 0, :, :] = data_cube[:, batch_list[i], :]
        label_output[i, :, :] = label_cube[:, batch_list[i], :]
    return data_output, label_output
def AE_batch_index(inline_number, xline_number):
    index_list = []
    for i in range(inline_number):
        index_list.append((0, i))
    for i in range(xline_number):
        index_list.append((1, i))
    return index_list
def random_batch_AE(data_cube, batch_list_input, z, mean, deviation):
    whole_index = batch_list_input * 1
    shuffle(whole_index)
    data_output = np.zeros(shape=(128, 1, 256, 352), dtype='float32')
    new_img = np.zeros(shape=(256, 352), dtype='float32')
    for i in range(128):
        # a = np.random.randint(0, 8)
        img_type = whole_index[i][0]
        img_index = whole_index[i][1]
        if img_type == 0:
            index_start = whole_index[i][2]
            data_img = input_data_norm(data_cube[:, img_index, :], mean, deviation)
            data_img = data_img[index_start*128:index_start*128+256, :]
            # phase_img = z[index_start*128:index_start*128+256, img_index, :]
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                # data_output[i, 1, :, :] = phase_img
                # data_output[i, 1, :, :] = z
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                # data_output[i, 1, :, :] = phase_img[::-1, :]
                # data_output[i, 1, :, :] = z[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                # data_output[i, 1, :, :] = phase_img[:, ::-1]
                # data_output[i, 1, :, :] = z[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                # data_output[i, 1, :, :] = phase_img[::-1, ::-1]
                # data_output[i, 1, :, :] = z[::-1, ::-1]
            # elif whole_index[i][3] == 4:
            #     a = data_img[:128, :]
            #     new_img[:128, :] = a
            #     new_img[128:, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            # elif whole_index[i][3] == 5:
            #     a = data_img[:128, :]
            #     new_img[128:, :] = a
            #     new_img[:128, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
            # elif whole_index[i][3] == 6:
            #     a = data_img[128:, :]
            #     new_img[:128, :] = a
            #     new_img[128:, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
            # elif whole_index[i][3] == 7:
            #     a = data_img[128:, :]
            #     new_img[128:, :] = a
            #     new_img[:128, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
        elif img_type == 1:
            index_start = whole_index[i][2]
            data_img = input_data_norm(data_cube[img_index, :, :], mean, deviation)
            data_img = data_img[index_start*136:index_start*136+256, :]
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                # data_output[i, 1, :, :] = phase_img
                # data_output[i, 1, :, :] = z
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                # data_output[i, 1, :, :] = phase_img[::-1, :]
                # data_output[i, 1, :, :] = z[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                # data_output[i, 1, :, :] = phase_img[:, ::-1]
                # data_output[i, 1, :, :] = z[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                # data_output[i, 1, :, :] = phase_img[::-1, ::-1]
                # data_output[i, 1, :, :] = z[::-1, ::-1]
            # elif whole_index[i][3] == 4:
            #     a = data_img[:128, :]
            #     new_img[:128, :] = a
            #     new_img[128:, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            # elif whole_index[i][3] == 5:
            #     a = data_img[:128, :]
            #     new_img[128:, :] = a
            #     new_img[:128, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
            # elif whole_index[i][3] == 6:
            #     a = data_img[128:, :]
            #     new_img[:128, :] = a
            #     new_img[128:, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
            # elif whole_index[i][3] == 7:
            #     a = data_img[128:, :]
            #     new_img[128:, :] = a
            #     new_img[:128, :] = a[::-1, :]
            #     data_output[i, 0, :, :] = new_img
            #     data_output[i, 1, :, :] = z
    return data_output
# def random_batch_FreezeUnet(data_cube, label_cube, batch_list_input):
#     # print("data_cube, label_cube shape:", data_cube.shape, label_cube.shape)
#     # shuffle(batch_list)
#     inline_index = [46, 196, 346]
#     whole_index = [46, 196, 346]
#     shuffle(whole_index)
#     inline_list = []
#     # sample_list = []
#     for i in range(32):
#     #     batch_list.append(np.random.randint(low=0, high=3))
#         inline_list.append(np.random.randint(low=0, high=4))
#     #     sample_list.append(np.random.randint(low=0, high=9))
#     # z_cube = np.zeros(shape=(1024, 392, 352))
#     # for i in range(352):
#     #     z_cube[:, :, i] = i/351
#     data_output = np.zeros(shape=(32, 1, 1024, 352), dtype='float32')
#     label_output = np.zeros(shape=(32, 1024, 352), dtype='float32')
#     for i in range(32):
#         a = np.random.randint(0, 3)
#         img_index = whole_index[a]
#         # data_img = data_cube[index_start:index_start+128, img_index, :]
#         data_img = input_data_norm(data_cube[:, img_index, :])
#         # data_img = data_img[index_start*320:index_start*320+384, :]
#         # data_img = input_data_norm(data_img)
#         # z_img = z_cube[index_start:index_start+392, img_index, :]
#         label_img = label_cube[:, img_index, :]
#         if inline_list[i] == 0:
#             data_output[i, 0, :, :] = data_img
#             # data_output[i, 1, :, :] = z_img
#             label_output[i, :, :] = label_img
#         elif inline_list[i] == 1:
#             data_output[i, 0, :, :] = data_img[::-1, :]
#             # data_output[i, 1, :, :] = z_img[::-1, :]
#             label_output[i, :, :] = label_img[::-1, :]
#         elif inline_list[i] == 2:
#             data_output[i, 0, :, :] = data_img[:, ::-1]
#             # data_output[i, 1, :, :] = z_img[::-1, :]
#             label_output[i, :, :] = label_img[:, ::-1]
#         elif inline_list[i] == 3:
#             data_output[i, 0, :, :] = data_img[::-1, ::-1]
#             # data_output[i, 1, :, :] = z_img[::-1, :]
#             label_output[i, :, :] = label_img[::-1, ::-1]
#     return data_output, label_output
def random_batch_FreezeUnet(data_cube, label_cube, batch_list_input, z, mean, deviation):
    whole_index = batch_list_input * 1
    shuffle(whole_index)
    data_output = np.zeros(shape=(124, 1, 256, 352), dtype='float32')
    label_output = np.zeros(shape=(124, 256, 352), dtype='float32')
    new_img = np.zeros(shape=(256, 352), dtype='float32')
    for i in range(124):
        # a = np.random.randint(0, 8)
        img_type = whole_index[i][0]
        img_index = whole_index[i][1]
        if img_type == 0:
            index_start = whole_index[i][2]
            data_img = input_data_norm(data_cube[:, img_index, :], mean, deviation)
            data_img = data_img[index_start:index_start + 256, :]
            # phase_img = z[index_start:index_start+256, img_index, :]
            label_img = label_cube[index_start:index_start + 256, img_index, :]
            # plt.imshow(label_img, cmap=plt.cm.rainbow)
            # plt.show()
            # data_output[i, 0, :, :] = data_img
            # data_output[i, 1, :, :] = z
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                # data_output[i, 1, :, :] = phase_img
                label_output[i, :, :] = label_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                # data_output[i, 1, :, :] = phase_img[::-1, :]
                label_output[i, :, :] = label_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                # data_output[i, 1, :, :] = phase_img[:, ::-1]
                label_output[i, :, :] = label_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                # data_output[i, 1, :, :] = phase_img[::-1, ::-1]
                label_output[i, :, :] = label_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:128, :]
                new_img[:128, :] = a
                new_img[128:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:128, :]
                new_img[:128, :] = b
                new_img[128:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:128, :]
                new_img[128:, :] = a
                new_img[:128, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:128, :]
                new_img[128:, :] = b
                new_img[:128, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[128:, :]
                new_img[:128, :] = a
                new_img[128:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[128:, :]
                new_img[:128, :] = b
                new_img[128:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[128:, :]
                new_img[128:, :] = a
                new_img[:128, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[128:, :]
                new_img[128:, :] = b
                new_img[:128, :] = b[::-1, :]
                label_output[i, :, :] = new_img
        elif img_type == 1:
            index_start = whole_index[i][2]
            data_img = input_data_norm(data_cube[img_index, :, :], mean, deviation)
            data_img = data_img[index_start:index_start + 256, :]
            # phase_img = z[img_index, index_start:index_start+256, :]
            label_img = label_cube[img_index, index_start:index_start + 256, :]
            # data_output[i, 1, :, :] = z
            if whole_index[i][3] == 0:
                data_output[i, 0, :, :] = data_img
                # data_output[i, 1, :, :] = phase_img
                label_output[i, :, :] = label_img
            elif whole_index[i][3] == 1:
                data_output[i, 0, :, :] = data_img[::-1, :]
                # data_output[i, 1, :, :] = phase_img[::-1, :]
                label_output[i, :, :] = label_img[::-1, :]
            elif whole_index[i][3] == 2:
                data_output[i, 0, :, :] = data_img[:, ::-1]
                # data_output[i, 1, :, :] = phase_img[:, ::-1]
                label_output[i, :, :] = label_img[:, ::-1]
            elif whole_index[i][3] == 3:
                data_output[i, 0, :, :] = data_img[::-1, ::-1]
                # data_output[i, 1, :, :] = phase_img[::-1, ::-1]
                label_output[i, :, :] = label_img[::-1, ::-1]
            elif whole_index[i][3] == 4:
                a = data_img[:128, :]
                new_img[:128, :] = a
                new_img[128:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:128, :]
                new_img[:128, :] = b
                new_img[128:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 5:
                a = data_img[:128, :]
                new_img[128:, :] = a
                new_img[:128, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[:128, :]
                new_img[128:, :] = b
                new_img[:128, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 6:
                a = data_img[128:, :]
                new_img[:128, :] = a
                new_img[128:, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[128:, :]
                new_img[:128, :] = b
                new_img[128:, :] = b[::-1, :]
                label_output[i, :, :] = new_img
            elif whole_index[i][3] == 7:
                a = data_img[128:, :]
                new_img[128:, :] = a
                new_img[:128, :] = a[::-1, :]
                data_output[i, 0, :, :] = new_img
                b = label_img[128:, :]
                new_img[128:, :] = b
                new_img[:128, :] = b[::-1, :]
                label_output[i, :, :] = new_img
    return data_output, label_output
def class_point_number(label_cube):
    inline_list = [46, 196, 346]
    xline_list = [142, 342, 542, 742, 942]
    img_range = 392
    class_point_counter = Counter()
    for i in range(3):
        for j in range(769):
            class_point_counter += Counter(label_cube[j:j+256, inline_list[i], :].flatten())
        print("class_point_counter:", class_point_counter, i)
    for i in range(5):
        for j in range(137):
            class_point_counter += Counter(label_cube[xline_list[i], j:j+256, :].flatten())
        print("class_point_counter:", class_point_counter, i)
    return
def batchlist(sample_number):
    batch_list = list(np.arange(sample_number))
    return batch_list
def label_count(label_cube, block_size=384, stride=10):
    print("label_cube shape:", label_cube.shape)
    output_counter = Counter()
    for i in range(label_cube.shape[1]):
        for j in range(int((label_cube.shape[0] - block_size + 1)//stride)):
            for k in range(int((label_cube.shape[2] - block_size + 1)//stride)):
                block = label_cube[j*10:j*10+256, i, k*10:k*10+256]
                output_counter += Counter(block.flatten())
                print(i, j, k)
    print("final output_counter:", output_counter)
    return
def input_data_norm(input_cube_img, mean, deviation):
    # input_cube_img_min = np.min(input_cube_img)
    # input_cube_img_max = np.max(input_cube_img)
    input_cube_img_norm = (input_cube_img - mean) / deviation
    return input_cube_img_norm
def model_train(mode):
    batch_list = batchlist(396)
    inline_list = batchlist(76)
    sample_list = batchlist(9)
    if mode == 'Normal':
        print('hello')
        torch.cuda.empty_cache()
        network = ResNet(n_classes=6)
        network.ParameterInitialize()
        total_params = sum(p.numel() for p in network.parameters())
        print('parameter number:\n', total_params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.566947901805945, 0.8277325486681695,
                                                                      1.1193825720203547, 0.7652958403078903,
                                                                      0.4886273204088382, 0.23201381678880156]).to(device),
                                            ignore_index=-1)
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + ' cards!')
            network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        network.to(device)
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.00291)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1)
        for state in optimizer.state.values():
            for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                pass
            pass
        pass
        loss_list = []
        data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
        data_cube = np.moveaxis(data_cube, 0, -1)
        data_cube = data_cube[0:1024, :, 21:]
        label_cube = np.load(r"label_3.npy")
        label_cube = label_cube[0:1024, :, 21:]
        print('Counter of label: ', Counter(label_cube.flatten()))
        for z in range(100):
            for i in range(100):
                torch.cuda.empty_cache()
                data, label = random_batch(data_cube=data_cube, label_cube=label_cube, batch_list=batch_list)
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
                label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
                # print(network.state_dict()['module.conv1.weight'])
                output = network(data)
                # label1, output1 = label_mask_and_select(label, output)
                # loss = MSE(output, label)
                loss = cross_entropy(output, label)
                print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
                loss_list.append(loss.cpu())
                loss.backward()
                loss = 0
                m = 0
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            if (z + 1) % 1 == 0:
                torch.save(network.state_dict(),
                           'saved_model' + str(int((z + 1) // 1)) + '.pt')  # 网络保存为saved_model.pt
                torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
        np.savetxt(r'loss_value2.txt', torch.Tensor(loss_list).detach().numpy())
    elif mode == 'AE':
        print('hello')
        torch.cuda.empty_cache()
        network = AutoEncoder(in_channels=1, out_channels=1)
        network.ParameterInitialize()
        total_params = sum(p.numel() for p in network.parameters())
        print('parameter number:\n', total_params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        MSE = nn.MSELoss(reduction='mean')
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + ' cards!')
            network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        network.to(device)
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0006)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9637, last_epoch=-1)
        for state in optimizer.state.values():
            for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                pass
            pass
        loss_list = []
        data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
        data_cube = np.moveaxis(data_cube, 0, -1)
        data_cube_mean = np.mean(data_cube)
        data_cube_deviation = np.var(data_cube)
        data_cube = data_cube[0:1024, :392, 21:373]
        mean = np.mean(data_cube)
        deviation = np.var(data_cube) ** 0.5
        # ins_phase = np.load(r"Ins_phase.npy")
        # label_cube = np.load(r"label_3.npy")
        # label_cube = label_cube[0:1024, :, 21:]
        # print('Counter of label: ', Counter(label_cube.flatten()))
        batch_list_I = batchlist(392)
        batch_list_X = batchlist(1024)
        batch_list1 = []
        for i in range(392):
            for j in range(7):
                for k in range(4):
                    batch_list1.append((0, i, j, k))
        for i in range(1024):
            for j in range(2):
                for k in range(4):
                    batch_list1.append((1, i, j, k))
        z_index = np.zeros(shape=(256, 352), dtype='float32')
        for i in range(352):
            z_index[:, i] = i*0.01 - 1.26
        for z in range(100):
            for i in range(80):
                torch.cuda.empty_cache()
                data = random_batch_AE(data_cube=data_cube, batch_list_input=batch_list1, z=1, mean=mean, deviation=deviation)
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
                # label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
                # print(network.state_dict()['module.conv1.weight'])
                output = network(data)
                # label1, output1 = label_mask_and_select(label, output)
                # loss = MSE(output, label)
                loss = MSE(output, data)
                print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
                loss_list.append(loss.cpu())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
            if (z + 1) % 10 == 0:
                torch.save(network.state_dict(),
                           'saved_modelAE' + str(int((z + 1) // 10)) + '.pt')  # 网络保存为saved_model.pt
                torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
        print('len(loss):', len(loss_list))
        np.savetxt(r'loss_valueAE.txt', torch.Tensor(loss_list).detach().numpy())
    elif mode == 'Freeze':
        print('hello')
        torch.cuda.empty_cache()
        network = FreezeUnet(in_channels=1, out_channels=6)
        network.ParameterInitialize()
        total_params = sum(p.numel() for p in network.parameters())
        print('parameter number:\n', total_params)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4410959118539384, 0.7977417345271601, 1.0732173989882963, 0.7387858454447427, 0.47772838894573333, 0.47143072024012955, 1]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.0053348416808006675, 1.1794575289895175, 1.1865171710991096, 1.1917482672384678, 1.2016214055722816, 1.2353207854198223]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9380593817231722, 0.3016548322264485, 0.40692383472537275, 0.2834924401887393, 0.18573017749836984, 0.18646788475400142, 4.697671448883896]).to(device),ignore_index=-1)
        # cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.765683352262653, 0.2483766316565284, 0.33474340556961707, 0.23163537767618422, 0.1506609242891989, 0.15104440805177177, 5.117855900494046]).to(device),ignore_index=-1)
        cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.4517990984262923, 0.7885890843557877, 1.062301109306234, 0.7362703457413214, 0.477581884703333, 0.48345847746703235]).to(device),ignore_index=-1)
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + ' cards!')
            network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        network.to(device)
        network.train()
        model_path = 'saved_modelAE10.pt'
        pre_weights = torch.load(model_path)
        print(pre_weights.keys())
        # print(pre_weights['left_conv_1.conv_ReLU.1.running_mean'])
        del_key_list = []
        del_keyword = ["module.conv2", "conv3", "conv4", "middle", "conv5", "conv6", "conv7", "conv8", "conv9"]
        for key, _ in pre_weights.items():
            for i in range(len(del_keyword)):
                if del_keyword[i] in key:
                    del_key_list.append(key)
            # elif '_3' in key:
            #     del_key_list.append(key)
        for key in del_key_list:
            del(pre_weights[key])
        # print(pre_weights.keys())
        print(pre_weights.keys())
        missing_keys, unexpected_keys = network.load_state_dict(pre_weights, strict=False)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0006)#0.00117
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.964, last_epoch=-1)#1, 0.975, -1
        for state in optimizer.state.values():
            for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
                if torch.is_tensor(v):
                    state[k] = v.cuda()
                pass
            pass
        loss_list = []
        data_cube = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
        data_cube = np.moveaxis(data_cube, 0, -1)
        data_cube = data_cube[0:1024, :392, 21:373]
        mean = np.mean(data_cube)
        deviation = np.var(data_cube) ** 0.5
        # data_cube_mean = np.mean(data_cube)
        # data_cube_deviation = np.var(data_cube)
        # data_cube = (data_cube - data_cube_mean) / data_cube_deviation
        # data_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        label_cube = np.load(r"label_3.npy")
        label_cube = label_cube[0:1024, :392, 21:373]
        print('label_cube shape:', label_cube.shape)
        class_point_number(label_cube)
        # ins_phase = np.load(r"Ins_phase.npy")

        # seismic_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        # label_cube_cut = np.zeros(shape=(data_cube.shape[0], 3, data_cube.shape[2]), dtype='float32')
        batch_list1 = []
        for i in range(3):
            for j in range(769):
                for k in range(4):
                    batch_list1.append((0, i*150+46, j, k))
        for i in range(5):
            for j in range(137):
                for k in range(4):
                    batch_list1.append((1, i*200+142, j, k))
        z_index = np.zeros(shape=(256, 352), dtype='float32')
        for i in range(352):
            z_index[:, i] = i * 0.01 - 1.26
        # print('Counter of label: ', Counter(label_cube.flatten()))
        # label_count(label_cube_cut)
        for z in range(100):#200
            for i in range(120):#40
                # print("module.conv1.conv1.weight:", network.state_dict()['module.conv1.conv1.weight'])
                # print("module.conv8.conv1.weight:", network.state_dict()['module.conv8.conv1.weight'])
                torch.cuda.empty_cache()
                data, label = random_batch_FreezeUnet(data_cube=data_cube, label_cube=label_cube, batch_list_input=batch_list1, z=1, mean=mean, deviation=deviation)
                data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
                label = (torch.autograd.Variable(torch.Tensor(label).long())).to(device)
                # print(network.state_dict()['module.conv1.conv2.weight'])
                output = network(data)
                # if z%10 == 0 and i % 1 == 0 and i>0:
                    # plt.cla()
                    # plt.clf()
                    # aba = torch.nn.Softmax(dim=1)(output)
                    # class_pos, class_no = torch.max(aba, 1, keepdim=True)
                    # model_output_index = torch.squeeze(class_no).cpu().detach().numpy()
                    # plt.imshow(model_output_index[0, :, :])
                    # plt.show()
                # label1, output1 = label_mask_and_select(label, output)
                # loss = MSE(output, label)
                loss = cross_entropy(output, label)
                print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
                loss_list.append(loss.cpu())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                output = 0
                data = 0
                label = 0
            scheduler.step()
            if (z + 1) % 10 == 0:
                torch.save(network.state_dict(),
                           'saved_modelFreeze' + str(int((z + 1) // 10)) + '.pt')  # 网络保存为saved_model.pt
                torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
        print('len(loss):', len(loss_list))
        np.savetxt(r'loss_valueFreeze.txt', torch.Tensor(loss_list).detach().numpy())
    return
if __name__ == '__main__':
    print('Start!')
    model_train(mode='Freeze')
    # model_train(mode='AE')
