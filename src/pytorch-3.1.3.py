import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
# 재사용을 위해 랜덤값을 초기화 합니다.
torch.manual_seed(1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

nb_epochs = 20000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수
    optimizer.zero_grad()
    cost.backward() # backward 연산
    optimizer.step() # 파라미터 업데이트

    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

print(list(model.parameters()))

new_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
with torch.no_grad():
    pred_y = model(new_var) # forward 연산
    print("훈련 후 입력이 4일 때의 예측값 :", pred_y)         