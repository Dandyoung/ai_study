# import torch


# python_lst = [1.2, 0.5, -0.7, 0.1]

# tensor = torch.tensor(python_lst, dtype=torch.float)
# print(tensor)
# print(tensor.shape)
# print(tensor.dim())


# # 즉, 차원은 중첩된 리스트의 깊이에 따라 결정되며, 주어진 예제의 경우 리스트가 2단계로 중첩되어 있어 2차원 텐서가 되는 것
# python_lst_2 = [[1.2, 0.5, -0.7, 0.1],
#              [1.2, 0.5, -0.7, 0.1],
#              [1.2, 0.5, -0.7, 0.1]]

# tensor_2 = torch.tensor(python_lst_2, dtype=torch.float)
# print(tensor_2)
# print(tensor_2.shape)
# print(tensor_2.dim())

# python_lst_3 = [[[1.2, 0.5, -0.7, 0.1],
#                  [1.2, 0.5, -0.7, 0.1],
#                  [1.2, 0.5, -0.7, 0.1]]]

# tensor_3 = torch.tensor(python_lst_3, dtype=torch.float)
# print(tensor_3)
# print(tensor_3.shape)
# print(tensor_3.dim())



import torch
print(torch.cuda.get_device_capability())  # compute capability 확인
print(torch.cuda.get_device_properties(0))  # 더 자세한 정보