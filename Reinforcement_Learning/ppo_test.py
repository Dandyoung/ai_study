# https://www.youtube.com/watch?v=l1CZQWBkdcY










class PPO :
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.gamma = 0.98
        # lmbda,eps, K는 클리핑 할때 사용?
        # lmbda : Generalized Advantage Estimation에 쓰이는 계수
        self.lmbda = 0.1
        self.eps = 0.1
        # K : 20스택동안 쌓은 데이터를 몇번 반복해서 학습할지 결정하는 것
        self.K = 2 # epoch 과 같은 의미

        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
    
    # 이 네트워크는 input dimension이 4이고, 
    def pi(self,x,softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self,x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, item):
        self.data.append(item)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for data in self.data :
            s,a,r,s_prime, prob_a, done = data
            # s는 그냥 niumpy array임
            s_lst.append(s)
            # a는 그냥 0 or 1 python 따라서 다른 a, r, done_mask, prob_a 같은 친구들도 dimension을 맞춰주기 위해 괄호 안에 넣어서 append를 해줌 (shape 안맞는다고 에러남)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime, prob_a, done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                           torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                           torch.tensor(prob_a_lst), torch.tensor(done_lst)
        self.data = []
        return s,a,r,s_prime, prob_a, done_mask


def main():
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=1)
    gamma = 0.99
    # T : 몇 time step 동안 data를 모을지, 어느 주기로 policy를 업데이트 할지 정하는 것.
    T = 20
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        # 처음 state 받고,
        s = env.reset()
        done = False
        # while문 안에 for문이 생김
        while not done :
            # T만큼 돌린 후 model.train() 호출
            for t in range(T) :
                
                prob = model.pi(torch.from_numpy(s).float())
                
                m = Categorical(prob)
                
                a = m.sample()
                
                s_prime, r, done, info = env.step(a)
                # prob[a].item() : 실제 내가 했던 action에 대한 확률 (4:6 이어도 4가 뽑힐 수 있음)
                # 나중에 ppo에서 ratio를 계산하는데 이떄 old policy 확률 값이 쓰인다.
                model.put_data((s,a,r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r 
                # 여기 break가 여기있는 이유는 27번 돌릴때, 20번 돌리고 나서, 남은 7번을 돌때, 20번 안돌고 7번만에 loop를 탈출시켜야하기에 안에
                if done : 
                    break
            model.train()

