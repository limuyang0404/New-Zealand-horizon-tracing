# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def check_inline(seismic_file, label_file, predict_file, inline_index):
    seismic_data = np.load(seismic_file)
    print('seismic_data shape:', seismic_data.shape)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    # seismic_data = np.moveaxis(seismic_data, 0, -1)
    seismic_data = seismic_data[0:1024, inline_index, 21:373]
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    label_data = np.load(label_file)
    print('label_data shape:', label_data.shape)
    label_data = label_data[0:1024, inline_index, 21:373]
    label_data = np.moveaxis(label_data, 0, -1)
    predict_data = np.load(predict_file)
    print('predict_data shape:', predict_data.shape)
    predict_data = predict_data[:, inline_index, :]
    predict_data = np.moveaxis(predict_data, 0, -1)
    alpha1 = np.ones(shape=label_data.shape, dtype='float32')
    alpha2 = np.ones(shape=label_data.shape, dtype='float32')
    alpha1[np.where(label_data<1)] = 0
    # alpha2[np.where(predict_data<1)] = 0
    # for i in range(479):
    #     for j in range(1024):
    #         if predict_data[i, j] != predict_data[i+1, j]:
    #             alpha2[i, j] = 1
    # plt.subplot(1, 2, 1)
    # plt.title('Label', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    # plt.imshow(label_data, alpha=alpha1)
    # plt.subplot(1, 2, 2)
    plt.title('Predict', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(predict_data, cmap=plt.cm.gray)
    plt.show()
    # plt.savefig("Label_46.png", dpi=300)
    return
def label_img_edit(img_path_list, label_file):
    label_cube = np.load(label_file)
    print('label_cube shape', label_cube.shape)
    print(label_cube.shape)
    for i in range(len(img_path_list)):
        iline_img = np.zeros(shape=(1043, 501))
        img_array = plt.imread(img_path_list[i])
        img_array = np.moveaxis(img_array, 0, 1)
        print('img_array shape:', img_array.shape)
        img_array = img_array * 255
        for j in range(1043):
            for k in range(501):
                if img_array[j, k, 0] == 64 and img_array[j, k, 1] == 67 and img_array[j, k, 2] == 135:
                    iline_img[j, k] = 1
                elif img_array[j, k, 0] == 41 and img_array[j, k, 1] == 120 and img_array[
                    j, k, 2] == 142:
                    iline_img[j, k] = 2
                elif img_array[j, k, 0] == 34 and img_array[j, k, 1] == 167 and img_array[
                    j, k, 2] == 132:
                    iline_img[j, k] = 3
                elif img_array[j, k, 0] == 121 and img_array[j, k, 1] == 209 and img_array[
                    j, k, 2] == 81:
                    iline_img[j, k] = 4
                elif img_array[j, k, 0] == 253 and img_array[j, k, 1] == 231 and img_array[
                    j, k, 2] == 36:
                    iline_img[j, k] = 5
        label_cube[:, i * 150 + 46, :] = iline_img
        # plt.show(img_array)
    np.save("label_2.npy", label_cube)

    return
def seismic_cube_cut(seismic_file):
    seismic_cube = np.load(seismic_file)
    seismic_cube = np.moveaxis(seismic_cube, 0, -1)
    print(seismic_cube.shape)
    seismic_cube = seismic_cube[:, :, :]
    np.save("seismic_2.npy", seismic_cube)
    return
# def check_inline(seismic_file, label_file, predict_file, inline_index):
#     seismic_data = np.load(seismic_file)
#     seismic_data = seismic_data[:, inline_index, :]
#     seismic_data = np.moveaxis(seismic_data, 0, -1)
#     label_data = np.load(label_file)
#     label_data = label_data[:, inline_index, :]
#     label_data = np.moveaxis(label_data, 0, -1)
#     predict_data = np.load(predict_file)
#     predict_data = predict_data[:, inline_index, :]
#     predict_data = np.moveaxis(predict_data, 0, -1)
#     alpha1 = np.ones(shape=label_data.shape, dtype='float32')
#     alpha2 = np.ones(shape=label_data.shape, dtype='float32')
#     alpha1[np.where(label_data<1)] = 0
#     alpha2[np.where(predict_data<1)] = 0
#     plt.subplot(1, 2, 1)
#     plt.title('Label', fontsize=18)
#     plt.imshow(seismic_data, cmap=plt.cm.gray)
#     plt.imshow(label_data, alpha=alpha1)
#     plt.subplot(1, 2, 2)
#     plt.title('Predict', fontsize=18)
#     plt.imshow(seismic_data, cmap=plt.cm.gray)
#     plt.imshow(predict_data, alpha=alpha2)
#     plt.show()
#     return
def line_horizon_trans_block(label_file):
    label_cube = np.load(label_file)#shape = (1043, 396, 501)
    # print("label_cube's shape", label_cube.shape)
    label_cube[689, 196, 183] = 4
    label_cube[742, 18, 37] = 1
    label_cube[942, 58, 37] = 1
    for i in range(1043):
        for j in range(396):
            trace_array = label_cube[i, j, :]
            trace_array = trace_trans(trace_array=trace_array)
            label_cube[i, j, :] = trace_array
            print(i, j)
    np.save(r"label_3.npy", label_cube)
    return
def block_add_fault(label_file):
    label_cube = np.load(label_file)
    seismic_data = np.load(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    print('seismic_data shape:', seismic_data.shape)
    seismic_data_mean = np.mean(seismic_data)
    seismic_data_deviation = np.var(seismic_data)
    seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - seismic_data_mean) / seismic_data_deviation
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print('seismic_data shape:', seismic_data.shape)
    seismic_data1 = seismic_data[342, :, :]
    seismic_data1 = np.moveaxis(seismic_data1, 0, -1)
    print('seismic_data shape:', seismic_data1.shape)
    print("label_cube's shape:", label_cube.shape)
    label_data = label_cube[142, :, :]
    label_data = np.moveaxis(label_data, 0, -1)
    fault_xline = np.load(r"Labels/fault/Fault_5Xline.npy")
    fault_inline = np.load(r"Labels/fault/Fault_3Inline.npy")
    print("fault_inline's shape:", fault_inline.shape)
    print("fault_xline's shape:", fault_xline.shape)
    alpha1 = np.ones(shape=(501, 396), dtype='float32')
    alpha2 = np.ones(shape=(501, 396), dtype='float32')
    # alpha1[np.where(fault_xline[:, 0, :] < 1)] = 0
    alpha2[np.where(label_data < 1)] = 0
    # plt.imshow(seismic_data1, cmap=plt.cm.gray)
    # # plt.imshow(label_data, alpha=alpha2)
    # plt.imshow(fault_xline[:, 1, :])
    # plt.show()
    fault_cube = np.zeros(shape=(1043, 396, 501), dtype='float32')

    for i in range(3):
        fault_cube[:, i*150+46, :] = np.moveaxis(fault_inline[:, :, i], 0, -1)
    for i in range(5):
        fault_cube[i*200+142, :, :] = np.moveaxis(fault_xline[:, i, :], 0, -1)
    label_cube[np.where(fault_cube > 0)] = 6
    # label_cube = label_cube + fault_cube
    plt.imshow(seismic_data1, cmap=plt.cm.gray)
    # plt.imshow(label_data, alpha=alpha2)
    plt.imshow(np.moveaxis(label_cube[942, :, :], 0, -1), alpha=alpha1)
    plt.show()
    np.save(r"label_5.npy", label_cube)
    return
def trace_trans(trace_array):
    output_trace_array = np.zeros(shape=trace_array.shape, dtype='float32')
    horizon_index = []
    for i in range(trace_array.shape[0]):
        if trace_array[i] > 0:
            horizon_index.append([trace_array[i], i])
    if len(horizon_index) > 0:
        for i in range(len(horizon_index)):
            output_trace_array[horizon_index[i][1]:] = horizon_index[i][0]
        # trace_array[:horizon_index[0]] = 0
        # trace_array[horizon_index[0]:horizon_index[1]] = 1
        # trace_array[horizon_index[1]:horizon_index[2]] = 2
        # trace_array[horizon_index[2]:horizon_index[3]] = 3
        # trace_array[horizon_index[3]:horizon_index[4]] = 4
        # trace_array[horizon_index[4]:] = 5
        return output_trace_array
    else:
        return trace_array
def check_inline_block(seismic_file, label_file, predict_file, inline_index):
    seismic_data = np.load(seismic_file)
    print("seismic_data shape:", seismic_data.shape)
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print("seismic_data shape:", seismic_data.shape)
    seismic_data = seismic_data[0:1024, :392, 21:373]
    print("seismic_data shape:", seismic_data.shape)
    mean = np.mean(seismic_data)
    deviation = np.var(seismic_data) ** 0.5
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = seismic_data[:, inline_index, :]
    print("seismic_data shape!:", seismic_data.shape)
    # seismic_data_mean = np.mean(seismic_data)
    # seismic_data_deviation = np.var(seismic_data)
    # seismic_data_deviation = seismic_data_deviation ** 0.5
    seismic_data = (seismic_data - mean) / deviation
    seismic_data = np.moveaxis(seismic_data, 0, -1)
    print('seismic min', np.min(seismic_data), np.max(seismic_data))
    label_data = np.load(label_file)
    print("label_data shape:", label_data.shape)
    label_data = label_data[0:1024, :392, 21:373]
    label_data = label_data[:, inline_index, :]
    label_data = np.moveaxis(label_data, 0, -1)
    predict_data = np.load(predict_file)
    # predict_data = predict_data[0:1024, :392, 21:373]
    predict_data = predict_data[:, inline_index, :]
    predict_data = np.moveaxis(predict_data, 0, -1)
    print('predict min', np.min(predict_data), np.max(predict_data))
    alpha1 = np.ones(shape=label_data.shape, dtype='float32')
    alpha2 = np.ones(shape=predict_data.shape, dtype='float32')
    # alpha1[np.where(label_data < 1)] = 0
    alpha2[np.where(predict_data < 1)] = 0
    blockedge = horizon_line(predict_data)
    # alpha2[np.where(blockedge < 1)] = 0
    print("seismic_data shape:", seismic_data.shape)
    print("predict_data shape:", predict_data.shape)
    plt.subplot(1, 2, 1)
    plt.title('Label', fontsize=18)
    plt.imshow(seismic_data, cmap=plt.cm.gray)
    # plt.imshow(label_data, cmap=plt.cm.rainbow)
    plt.subplot(1, 2, 2)
    plt.title('Predict', fontsize=18)
    # plt.imshow(seismic_data, cmap=plt.cm.gray)
    plt.imshow(predict_data, cmap=plt.cm.rainbow)

    # plt.imsave(r'/home/limuyang/New_zealand_data/manual_interpretation/' + '0ilineblock.png', predict_data)
    # plt.imshow(blockedge)
    plt.show()
    return
def block_edge(predict_result):
    output = np.zeros(shape=(predict_result.shape[0], predict_result.shape[1], 4), dtype='float32')
    r = [0, 0, 0, 1, 1, 1]
    g = [0, 1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 1, 0]
    for i in range(predict_result.shape[0]-1):
        for j in range(predict_result.shape[1]):
            if predict_result[i, j] == predict_result[i+1, j]-1:
                output[i, j, 0] = r[int(predict_result[i, j])]
                output[i, j, 1] = g[int(predict_result[i, j])]
                output[i, j, 2] = b[int(predict_result[i, j])]
                output[i, j, 3] = 1
    return output
def horizon_line(predict_result):
    print('predict_result shape', predict_result.shape)
    output = np.zeros(shape=(predict_result.shape[0], predict_result.shape[1], 4), dtype='float32')
    output1 = np.zeros(shape=predict_result.shape, dtype='float32')
    output2 = np.zeros(shape=predict_result.shape, dtype='float32')
    output3 = np.zeros(shape=predict_result.shape, dtype='float32')
    output4 = np.zeros(shape=predict_result.shape, dtype='float32')
    r = [0, 0, 0, 1, 1, 1, 1]
    g = [0, 1, 1, 0, 0, 1, 1]
    b = [1, 0, 1, 0, 1, 0, 1]
    for i in range(7):
        a = np.where(predict_result == i)
        # print('a:', a, a[1].shape, np.max(a[0]), np.max(a[1]))
        output1[np.where(predict_result == i)] = r[i]
        output2[np.where(predict_result == i)] = g[i]
        output3[np.where(predict_result == i)] = b[i]
        output4[np.where(predict_result == i)] = 1
    output[:, :, 0] = output1
    output[:, :, 1] = output2
    output[:, :, 2] = output3
    output[:, :, 3] = output4
    return output
if __name__=='__main__':
    print('hello!')
    # check_inline(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_2.npy", r"predictAE.npy", 100)
    # label_img_edit(['0_iline.png', '1_iline.png', '2_iline.png'], 'label.npy')
    # seismic_cube_cut(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy")
    # check_inline(r"seismic_2.npy", r"label_2.npy", r"predictAE.npy", 46)
    # line_horizon_trans_block("label.npy")
    # check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_4.npy", r"label_4.npy", 46)
    check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_3.npy", r"predictFreeze.npy", 100)
    # block_add_fault("label_2.npy")
    # check_inline_block(r"/home/limuyang/New_zealand_data/Seismic/Opunake_Quad_A.npy", r"label_4.npy", r"Ins_phase.npy", 46)
