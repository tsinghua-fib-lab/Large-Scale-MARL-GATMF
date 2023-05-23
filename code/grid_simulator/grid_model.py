import numpy as np

class Model:
    def __init__(self):
        pass

    def init_exogenous_variables(self,num_grid,num_diamond,diamond_extent,num_move):
        self.num_grid=num_grid
        self.num_diamond=num_diamond
        self.diamond_extent=diamond_extent
        self.num_move=num_move
        self.field_length=3*self.num_grid

        x,y=np.meshgrid(range(-3*self.diamond_extent-1,3*(self.diamond_extent)+2),range(-3*self.diamond_extent-1,3*(self.diamond_extent)+2))
        self.diamond_mat=np.exp(-2*((x/3)**2+(y/3)**2)/self.field_length)

        self.Gmat=np.zeros((self.num_grid*self.num_grid,self.num_grid*self.num_grid))
        for x in range(self.num_grid):
            for y in range(self.num_grid):
                pos=self.num_grid*x+y
                self.Gmat[pos,self.num_grid*((x-1)%self.num_grid)+y%self.num_grid]=1
                self.Gmat[pos,self.num_grid*((x+1)%self.num_grid)+y%self.num_grid]=1
                self.Gmat[pos,self.num_grid*(x%self.num_grid)+(y-1)%self.num_grid]=1
                self.Gmat[pos,self.num_grid*(x%self.num_grid)+(y+1)%self.num_grid]=1

    def init_endogenous_variables(self):
        self.field=np.zeros((self.field_length,self.field_length))
        self.miner=np.ones((self.num_grid,self.num_grid))
        self.diamond_pos=list()

        for i in range(self.num_diamond):
            rand_x=np.random.randint(0,self.num_grid)
            rand_y=np.random.randint(0,self.num_grid)
            self.diamond_pos.append([rand_x,rand_y])

            for x in range(-self.diamond_extent,self.diamond_extent+1):
                for y in range(-self.diamond_extent,self.diamond_extent+1):
                    self.field[(3*(rand_x+x))%self.field_length,(3*(rand_y+y))%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x),3*(self.diamond_extent+y)]
                    self.field[(3*(rand_x+x))%self.field_length,(3*(rand_y+y)+1)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x),3*(self.diamond_extent+y)+1]
                    self.field[(3*(rand_x+x))%self.field_length,(3*(rand_y+y)+2)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x),3*(self.diamond_extent+y)+2]

                    self.field[(3*(rand_x+x)+1)%self.field_length,(3*(rand_y+y))%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+1,3*(self.diamond_extent+y)]
                    self.field[(3*(rand_x+x)+1)%self.field_length,(3*(rand_y+y)+1)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+1,3*(self.diamond_extent+y)+1]
                    self.field[(3*(rand_x+x)+1)%self.field_length,(3*(rand_y+y)+2)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+1,3*(self.diamond_extent+y)+2]

                    self.field[(3*(rand_x+x)+2)%self.field_length,(3*(rand_y+y))%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+2,3*(self.diamond_extent+y)]
                    self.field[(3*(rand_x+x)+2)%self.field_length,(3*(rand_y+y)+1)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+2,3*(self.diamond_extent+y)+1]
                    self.field[(3*(rand_x+x)+2)%self.field_length,(3*(rand_y+y)+2)%self.field_length]+=self.diamond_mat[3*(self.diamond_extent+x)+2,3*(self.diamond_extent+y)+2]

        self.init_return=self.get_return()

    def move_miner(self,action_vector):
        action_vector=action_vector.reshape(self.num_grid,self.num_grid,-1)
        move_vector=self.num_move*action_vector
        self.miner=self.miner-self.num_move

        self.miner[:-1,:]+=move_vector[1:,:,0]
        self.miner[-1,:]+=move_vector[0,:,0]

        self.miner[1:,:]+=move_vector[:-1,:,1]
        self.miner[0,:]+=move_vector[-1,:,1]

        self.miner[:,:-1]+=move_vector[:,1:,2]
        self.miner[:,-1]+=move_vector[:,0,2]

        self.miner[:,1:]+=move_vector[:,:-1,3]
        self.miner[:,0]+=move_vector[:,-1,3]

        self.miner=np.clip(self.miner,0,None)

    def output_record(self):
        left_up=self.field[::3,::3].reshape(self.num_grid*self.num_grid,-1)
        right_up=self.field[::3,2::3].reshape(self.num_grid*self.num_grid,-1)
        center=self.field[1::3,1::3].reshape(self.num_grid*self.num_grid,-1)
        left_down=self.field[2::3,::3].reshape(self.num_grid*self.num_grid,-1)
        right_down=self.field[2::3,2::3].reshape(self.num_grid*self.num_grid,-1)

        return np.hstack((left_up,right_up,center,left_down,right_down))

    def get_reward(self):
        reward=np.empty((self.num_grid,self.num_grid))
        for x in range(self.num_grid):
            for y in range(self.num_grid):
                reward[x,y]=self.miner[x,y]*np.sum(self.field[3*x:3*x+3,3*y:3*y+3])

        reward_sum=np.empty((self.num_grid,self.num_grid))
        for x in range(self.num_grid):
            for y in range(self.num_grid):
                reward_sum[x,y]=reward[(x-1)%self.num_grid,y]+reward[(x+1)%self.num_grid,y]+reward[x,(y-1)%self.num_grid]+reward[x,(y+1)%self.num_grid]

        return reward_sum.reshape(-1,1)

    def get_return(self):
        reward=np.empty((self.num_grid,self.num_grid))
        for x in range(self.num_grid):
            for y in range(self.num_grid):
                reward[x,y]=self.miner[x,y]*np.sum(self.field[3*x:3*x+3,3*y:3*y+3])

        return np.sum(reward)

    def show_grid(self,path):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9,6))
        plt.contour(self.field.T)
        cb=plt.colorbar()
        cb.set_label(label='Number of Diamonds',size=30)
        cb.ax.tick_params(labelsize=20)

        plt.imshow(self.miner.repeat(3,0).repeat(3,1),cmap='Reds',vmin=0.3,vmax=1.7)
        cb=plt.colorbar()
        cb.set_label(label='Number of Miners',size=30)
        cb.ax.tick_params(labelsize=20)

        for diamond in self.diamond_pos:
            plt.scatter(3*diamond[0]+1,3*diamond[1]+1,marker='d',facecolor='Blue')
        plt.hlines(np.linspace(2.5,self.field_length-3.5,self.num_grid-1),xmin=-0.5,xmax=self.field_length-0.5,color='red',linewidth=0.3)
        plt.vlines(np.linspace(2.5,self.field_length-3.5,self.num_grid-1),ymin=-0.5,ymax=self.field_length-0.5,color='red',linewidth=0.3)
        plt.xticks(range(1,self.field_length+1,3),range(0,self.num_grid),fontsize=20)
        plt.yticks(range(1,self.field_length+1,3),range(0,self.num_grid),fontsize=20)

        plt.title('Return=%.2f'%(self.get_return()-self.init_return),fontsize=30)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        print(self.get_return()-self.init_return,np.sum(self.miner))