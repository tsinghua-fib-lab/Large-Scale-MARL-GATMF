# environment setting
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])

# external packages
import numpy as np
import time
import torch
import copy
import json
from tqdm import tqdm

# self writing files
import networks as networks
from reward_fun import reward_fun
import constants
import data_range
from simulator import disease_model
from util import *

class MARL(object):
    def __init__(self,
                MSA_name='Atlanta',
                vaccine_day=0.01,
                step_length=24,
                num_seed=64,
                max_episode=110,
                update_batch=300,
                batch_size=24,
                buffer_capacity=64,
                save_interval=1,
                lr=0.0001,
                lr_decay=False,
                grad_clip=False,
                max_grad_norm=10,
                soft_replace_rate=0.01,
                gamma=0.6,
                D_weight=0,
                entropy_weight=0,
                reward_option=0,
                explore_noise=0.002,
                explore_noise_decay=True,
                explore_noise_decay_rate=0.2,
                manual_seed=0,
                random_simulation=False):
        super().__init__()

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.manual_seed=manual_seed
        torch.manual_seed(self.manual_seed)
        if self.device=='cuda':
            torch.cuda.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.random_simulation=random_simulation

        # loading cbg data (for simulation)
        self.MSA_name=MSA_name
        self.data_range_key=(self.MSA_name+'_random') if self.random_simulation else self.MSA_name
        self.data=load_data(self.MSA_name)
        self.poi_areas=self.data['poi_areas']
        self.cbg_sizes=self.data['cbg_sizes']
        self.poi_cbg_visits_list=self.data['poi_cbg_visits_list']# time_length*poi*cbg
        self.time_length=len(self.poi_cbg_visits_list)
        self.day_length=int(self.time_length/24)
        self.step_length=step_length

        self.num_cbg=len(self.cbg_sizes)
        self.sum_population=np.sum(self.cbg_sizes)
        assert len(self.poi_areas)==self.poi_cbg_visits_list[0].shape[0]
        self.num_poi=len(self.poi_areas)

        # simulator
        self.num_seed=num_seed
        self.simulator=disease_model.Model(num_seeds=self.num_seed,random_simulation=self.random_simulation)
        self.simulator.init_exogenous_variables(poi_areas=self.poi_areas,
                                poi_dwell_time_correction_factors=self.data['poi_dwell_time_correction_factors'],
                                cbg_sizes=self.cbg_sizes,
                                poi_cbg_visits_list=self.poi_cbg_visits_list,
                                cbg_attack_rates_original = self.data['cbg_attack_rates_scaled'],
                                cbg_death_rates_original = self.data['cbg_death_rates_scaled'],
                                p_sick_at_t0=constants.parameters_dict[self.MSA_name][0],
                                home_beta=constants.parameters_dict[self.MSA_name][1],
                                poi_psi=constants.parameters_dict[self.MSA_name][2],
                                just_compute_r0=False,
                                latency_period=96,  # 4 days
                                infectious_period=84,  # 3.5 days
                                confirmation_rate=.1,
                                confirmation_lag=168,  # 7 days
                                death_lag=432)

        # adjency matrix
        self.Gmat=np.load(os.path.join('..','data',self.MSA_name,f'{self.MSA_name}_Gmat.npy'))
        self.Gmat=np.clip(self.Gmat,0,data_range.Gmat[self.data_range_key])/data_range.Gmat[self.data_range_key]
        self.Gmat=torch.FloatTensor(self.Gmat).to(self.device)

        # dynamic features
        self.cbg_state_raw=np.zeros((self.num_seed,self.num_cbg,3),dtype=np.float32)#S,C,D
        self.cbg_state=np.zeros((self.num_seed,self.num_cbg,3),dtype=np.float32)#S,C,D
        self.cbg_state_diff=np.zeros((self.num_seed,self.num_cbg,3),dtype=np.float32)

        # vaccine number
        self.vaccine_day=int(vaccine_day*self.sum_population)

        # learning parameters
        self.max_episode=max_episode
        self.update_batch=update_batch
        self.batch_size=batch_size
        self.save_interval=save_interval
        self.lr=lr
        self.lr_decay=lr_decay
        self.grad_clip=grad_clip
        self.max_grad_norm=max_grad_norm
        self.soft_replace_rate=soft_replace_rate
        self.gamma=gamma
        self.D_weight=D_weight
        self.entropy_weight=entropy_weight
        self.reward_option=reward_option
        self.explore_noise=explore_noise
        self.explore_noise_decay=explore_noise_decay
        self.explore_noise_decay_rate=explore_noise_decay_rate

        # networks and optimizers
        self.actor=networks.Actor().to(self.device)
        self.actor_target=copy.deepcopy(self.actor).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.lr)

        self.critic=networks.Critic().to(self.device)
        self.critic_target=copy.deepcopy(self.critic).eval()
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=self.lr)

        self.actor_attention=networks.Attention().to(self.device)
        self.actor_attention_target=copy.deepcopy(self.actor_attention).eval()
        self.actor_attention_optimizer=torch.optim.Adam(self.actor_attention.parameters(),lr=self.lr)

        self.critic_attention=networks.Attention().to(self.device)
        self.critic_attention_target=copy.deepcopy(self.critic_attention).eval()
        self.critic_attention_optimizer=torch.optim.Adam(self.critic_attention.parameters(),lr=self.lr)

        if self.lr_decay:
            self.actor_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.critic_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.actor_attention_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_attention_optimizer, lr_lambda=lambda epoch: 0.99**epoch)
            self.critic_attention_optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_attention_optimizer, lr_lambda=lambda epoch: 0.99**epoch)

        # buffer
        self.buffer_capacity=self.num_seed*(self.day_length-1)*buffer_capacity
        self.buffer_pointer=0
        self.buffer_size=0
        self.buffer_s=np.empty((self.buffer_capacity,self.num_cbg,6),dtype=np.float32) # S,C,D,Sdiff,Cdiff,Ddiff
        self.buffer_a=np.empty((self.buffer_capacity,self.num_cbg,1),dtype=np.float32)
        self.buffer_s1=np.empty((self.buffer_capacity,self.num_cbg,6),dtype=np.float32) # S,C,D,Sdiff,Cdiff,Ddiff
        self.buffer_r=np.empty((self.buffer_capacity,self.num_cbg,1),dtype=np.float32)
        self.buffer_end=np.ones((self.buffer_capacity,self.num_cbg,1),dtype=np.float32)

        # training trackors
        self.episode_deaths_trackor=list()
        self.episode_cases_trackor=list()
        self.critic_loss_trackor=list()
        self.actor_loss_trackor=list()
        self.action_entropy_trackor=list()

        # making output directory
        self.output_dir=os.path.join('..','model',f'{self.MSA_name}_{self.num_seed}seeds_{time_data}')
        os.mkdir(self.output_dir)
        with open(os.path.join(self.output_dir,'config.json'),'w') as f:
            json.dump(config_data,f)

        print(f'Training platform on {self.MSA_name} initialized')
        print(f'Number of CBGs={self.num_cbg}')
        print(f'Number of POIs={self.num_poi}')
        print(f'Total population={self.sum_population}')
        print(f'Time length={self.time_length}')
        print(f'Train with {self.num_seed} random seeds')

    def test_simulation(self):
        for num in range(1):
            self.simulator.reset_random_seed()
            self.simulator.init_endogenous_variables()
            # mat=500*np.ones((self.num_seed,self.num_cbg))
            # mat[:30,:]-=300
            # self.simulator.add_vaccine(mat)
            for i in tqdm(range(63)):
                # if i==20:
                #     mat=500*np.ones((self.num_seed,self.num_cbg))
                #     self.simulator.add_vaccine(mat)
                self.simulator.simulate_disease_spread(no_print=True)
                # current_C,current_D=self.simulator.output_record()

            T1,L_1,I_1,R_1,C2,D2,total_affected, history_C2, history_D2, total_affected_each_cbg=self.simulator.output_record(full=True)
            print(np.mean(np.sum(history_C2[:,-1,:],axis=-1)))
            print(np.mean(np.sum(history_D2[:,-1,:],axis=-1)))

            print(history_C2[:5,-1,:10])

            gt_result_root=os.path.join('..','model','simulator_test')
            if not os.path.exists(gt_result_root):
                os.mkdir(gt_result_root)
            savepath = os.path.join(gt_result_root, f'cases_cbg_no_vaccination_{self.MSA_name}_{self.num_seed}seeds_step_raw{num}.npy')
            np.save(savepath, history_C2)
            savepath = os.path.join(gt_result_root, f'deaths_cbg_no_vaccination_{self.MSA_name}_{self.num_seed}seeds_step_raw{num}.npy')
            np.save(savepath, history_D2)

    def test_network(self):
        index=np.array(range(self.batch_size))
        state=torch.FloatTensor(self.buffer_s[index,:,:]).to(self.device)
        print(state.shape)

        Actor_attention=self.actor_attention(state,self.Gmat)
        print(Actor_attention.shape)
        Actor_state_bar=torch.bmm(Actor_attention,state)
        print(Actor_state_bar.shape)
        Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
        print(Actor_state_all.shape)
        action=self.actor(Actor_state_all)
        print(action.shape)

        Critic_attention=self.critic_attention(state,self.Gmat)
        print(Critic_attention.shape)
        Critic_state_bar=torch.bmm(Critic_attention,state)
        print(Critic_state_bar.shape)
        Critic_state_all=torch.concat([state,Critic_state_bar],dim=-1)
        print(Critic_state_all.shape)
        action_bar=torch.bmm(Critic_attention,action)
        print(action_bar.shape)
        action_all=torch.concat([action,action_bar],dim=-1)
        print(action_all.shape)
        estimate_reward=self.critic(Critic_state_all,action_all)
        print(estimate_reward.shape)

    def update_cbg_state(self,current_C,current_D):
        self.cbg_state_raw[:,:,1]=current_C
        self.cbg_state_raw[:,:,2]=current_D
        self.cbg_state_raw[:,:,0]=self.cbg_sizes-current_C-current_D

        self.cbg_state[:,:,1]=np.clip(self.cbg_state_raw[:,:,1],0,data_range.Idata[self.data_range_key])/data_range.Idata[self.data_range_key]
        self.cbg_state[:,:,2]=np.clip(self.cbg_state_raw[:,:,2],0,data_range.Ddata[self.data_range_key])/data_range.Ddata[self.data_range_key]
        self.cbg_state[:,:,0]=np.clip(self.cbg_state_raw[:,:,0],0,data_range.cbg_size[self.data_range_key])/data_range.cbg_size[self.data_range_key]

    def norm_cbg_state_diff(self):
        self.cbg_state_diff[:,:,1]=np.clip(self.cbg_state_diff[:,:,1],-data_range.Idata_diff[self.data_range_key],data_range.Idata_diff[self.data_range_key])/data_range.Idata_diff[self.data_range_key]
        self.cbg_state_diff[:,:,2]=np.clip(self.cbg_state_diff[:,:,2],-data_range.Ddata_diff[self.data_range_key],data_range.Ddata_diff[self.data_range_key])/data_range.Ddata_diff[self.data_range_key]
        self.cbg_state_diff[:,:,0]=np.clip(self.cbg_state_diff[:,:,0],-data_range.Sdata_diff[self.data_range_key],data_range.Sdata_diff[self.data_range_key])/data_range.Sdata_diff[self.data_range_key]

    def get_vaccine(self,action):
        weight=torch.softmax(action,dim=1)
        vaccine_mat=self.vaccine_day*weight

        return vaccine_mat

    def get_entropy(self,action):
        weight=torch.softmax(action,dim=1)
        action_entropy=-torch.sum(weight*torch.log2(weight))

        return action_entropy

    def update(self):
        actor_loss_sum=0
        action_entropy_sum=0
        critic_loss_sum=0

        for batch_count in range(self.update_batch):
            batch_index=np.random.randint(self.buffer_size,size=self.batch_size)
            s_batch=torch.FloatTensor(self.buffer_s[batch_index]).to(self.device)
            a_batch=torch.FloatTensor(self.buffer_a[batch_index]).to(self.device)
            s1_batch=torch.FloatTensor(self.buffer_s1[batch_index]).to(self.device)
            r_batch=torch.FloatTensor(self.buffer_r[batch_index]).to(self.device)
            end_batch=torch.FloatTensor(self.buffer_end[batch_index]).to(self.device)

            with torch.no_grad():
                update_Actor_attention1=self.actor_attention_target(s1_batch,self.Gmat)
                update_Actor_state1_bar=torch.bmm(update_Actor_attention1,s1_batch)
                update_Actor_state1_all=torch.concat([s1_batch,update_Actor_state1_bar],dim=-1)
                update_action1=self.actor_target(update_Actor_state1_all)
                # update_action1=self.get_vaccine(update_action1)/self.vaccine_day
                # update_action1=self.get_vaccine(update_action1)/(2*self.vaccine_day/self.num_cbg)
                # update_action1=self.get_vaccine(update_action1)

                update_Critic_attention1=self.critic_attention_target(s1_batch,self.Gmat)
                update_Critic_state1_bar=torch.bmm(update_Critic_attention1,s1_batch)
                update_Critic_state1_all=torch.concat([s1_batch,update_Critic_state1_bar],dim=-1)
                update_action1_bar=torch.bmm(update_Critic_attention1,update_action1)
                update_action1_all=torch.concat([update_action1,update_action1_bar],dim=-1)

                y=r_batch+self.gamma*self.critic_target(update_Critic_state1_all,update_action1_all)*end_batch

            update_Critic_attention=self.critic_attention(s_batch,self.Gmat)
            update_Critic_state_bar=torch.bmm(update_Critic_attention,s_batch)
            update_Critic_state_all=torch.concat([s_batch,update_Critic_state_bar],dim=-1)
            update_action_bar=torch.bmm(update_Critic_attention,a_batch)
            update_action_all=torch.concat([a_batch,update_action_bar],dim=-1)

            critic_loss=torch.sum(torch.square(y-self.critic(update_Critic_state_all,update_action_all)))/self.batch_size
            critic_loss_sum+=critic_loss.cpu().item()

            self.critic_optimizer.zero_grad()
            self.critic_attention_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_attention.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_attention_optimizer.step()

            update_Actor_attention=self.actor_attention(s_batch,self.Gmat)
            update_Actor_state_bar=torch.bmm(update_Actor_attention,s_batch)
            update_Actor_state_all=torch.concat([s_batch,update_Actor_state_bar],dim=-1)
            update_action=self.actor(update_Actor_state_all)
            action_entropy=self.get_entropy(update_action)
            action_entropy_sum+=action_entropy.cpu().item()
            # update_action=self.get_vaccine(update_action)/self.vaccine_day
            # update_action=self.get_vaccine(update_action)/(2*self.vaccine_day/self.num_cbg)
            # update_action=self.get_vaccine(update_action)

            with torch.no_grad():
                update_Critic_attention_new=self.critic_attention(s_batch,self.Gmat)
                update_Critic_state_bar_new=torch.bmm(update_Critic_attention_new,s_batch)
                update_Critic_state_all_new=torch.concat([s_batch,update_Critic_state_bar_new],dim=-1)

            update_action_bar_new=torch.bmm(update_Critic_attention_new,update_action)
            update_action_all_new=torch.concat([update_action,update_action_bar_new],dim=-1)

            # print(action_entropy)
            # print(-torch.sum(self.critic(update_Critic_state_all_new,update_action_all_new)))

            actor_loss=-torch.sum(self.critic(update_Critic_state_all_new,update_action_all_new))/self.batch_size-self.entropy_weight*action_entropy/self.batch_size
            actor_loss_sum+=actor_loss.cpu().item()

            self.actor_optimizer.zero_grad()
            self.actor_attention_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_attention.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_attention_optimizer.step()

            for x in self.actor.state_dict().keys():
                eval('self.actor_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.actor_target.'+x+'.data.add_(self.soft_replace_rate*self.actor.'+x+'.data)')
            for x in self.actor_attention.state_dict().keys():
                eval('self.actor_attention_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.actor_attention_target.'+x+'.data.add_(self.soft_replace_rate*self.actor_attention.'+x+'.data)')
            for x in self.critic.state_dict().keys():
                eval('self.critic_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.critic_target.'+x+'.data.add_(self.soft_replace_rate*self.critic.'+x+'.data)')
            for x in self.critic_attention.state_dict().keys():
                eval('self.critic_attention_target.'+x+'.data.mul_(1-self.soft_replace_rate)')
                eval('self.critic_attention_target.'+x+'.data.add_(self.soft_replace_rate*self.critic_attention.'+x+'.data)')

        if self.lr_decay:
            self.actor_optimizer_scheduler.step()
            self.actor_attention_optimizer_scheduler.step()
            self.critic_optimizer_scheduler.step()
            self.critic_attention_optimizer_scheduler.step()

        critic_loss_mean=critic_loss_sum/self.update_batch
        self.critic_loss_trackor.append(critic_loss_mean)
        actor_loss_mean=actor_loss_sum/self.update_batch
        self.actor_loss_trackor.append(actor_loss_mean)
        action_entropy_mean=action_entropy_sum/self.update_batch
        self.action_entropy_trackor.append(action_entropy_mean)

        tqdm.write(f'Update: Critic Loss {critic_loss_mean} | Actor Loss {actor_loss_mean} | Action Entropy {action_entropy_mean}')

    def train(self):
        for episode in tqdm(range(self.max_episode)):
            self.simulator.init_endogenous_variables()
            self.update_cbg_state(0,0)
            self.cbg_state_init=copy.deepcopy(self.cbg_state_raw)

            self.simulator.simulate_disease_spread(length=self.step_length,verbosity=1,no_print=True)
            current_C,current_D=self.simulator.output_record()
            self.simulator.empty_record()

            self.update_cbg_state(current_C,current_D)
            self.cbg_state_diff=self.cbg_state_raw-self.cbg_state_init
            self.norm_cbg_state_diff()

            if self.explore_noise_decay:
                self.explore_noise=self.explore_noise/((episode+1)**self.explore_noise_decay_rate)

            for day in range(1,self.day_length):
                current_state=np.concatenate((self.cbg_state,self.cbg_state_diff),axis=-1)
                self.buffer_s[self.buffer_pointer:self.buffer_pointer+self.num_seed]=current_state

                with torch.no_grad():
                    state=torch.FloatTensor(current_state).to(self.device)

                    Actor_attention=self.actor_attention(state,self.Gmat)
                    Actor_state_bar=torch.bmm(Actor_attention,state)
                    Actor_state_all=torch.concat([state,Actor_state_bar],dim=-1)
                    action=self.actor(Actor_state_all)

                    action=action+torch.randn_like(action)*self.explore_noise
                    vaccine_mat=self.get_vaccine(action).cpu().numpy()

                self.simulator.add_vaccine(vaccine_mat.squeeze(-1))
                self.simulator.simulate_disease_spread(length=self.step_length,verbosity=1,no_print=True)
                current_C,current_D=self.simulator.output_record()
                self.simulator.empty_record()

                cbg_state_raw_old=copy.deepcopy(self.cbg_state_raw)
                self.update_cbg_state(current_C,current_D)
                self.cbg_state_diff=self.cbg_state_raw-cbg_state_raw_old
                self.norm_cbg_state_diff()
                reward=reward_fun(cbg_state_raw_old,self.cbg_state_raw,D_weight=self.D_weight,option=self.reward_option)

                self.buffer_s1[self.buffer_pointer:self.buffer_pointer+self.num_seed]=np.concatenate((self.cbg_state,self.cbg_state_diff),axis=-1)
                # self.buffer_a[self.buffer_pointer:self.buffer_pointer+self.num_seed]=vaccine_mat/self.vaccine_day
                # self.buffer_a[self.buffer_pointer:self.buffer_pointer+self.num_seed]=vaccine_mat/(2*self.vaccine_day/self.num_cbg)
                # self.buffer_a[self.buffer_pointer:self.buffer_pointer+self.num_seed]=vaccine_mat
                self.buffer_a[self.buffer_pointer:self.buffer_pointer+self.num_seed]=action.cpu().numpy()
                self.buffer_r[self.buffer_pointer:self.buffer_pointer+self.num_seed]=reward
                if day==self.day_length-1:
                    self.buffer_end[self.buffer_pointer:self.buffer_pointer+self.num_seed]=0

                self.buffer_size=max(self.buffer_size,self.buffer_pointer+self.num_seed)
                self.buffer_pointer=(self.buffer_pointer+self.num_seed)%self.buffer_capacity

            episode_C=np.mean(np.sum(self.cbg_state_raw[:,:,1],axis=1),axis=0)
            self.episode_cases_trackor.append(episode_C)
            episode_D=np.mean(np.sum(self.cbg_state_raw[:,:,2],axis=1),axis=0)
            self.episode_deaths_trackor.append(episode_D)

            tqdm.write(f'Episode {episode}: Cases {episode_C} | Deaths {episode_D}')

            self.update()

            if (episode+1)%self.save_interval==0:
                self.save_models(episode+1)

    def save_models(self,episode):
        torch.save(self.actor.state_dict(),os.path.join(self.output_dir,f'actor_{episode}.pth'))
        torch.save(self.actor_target.state_dict(),os.path.join(self.output_dir,f'actor_target_{episode}.pth'))

        torch.save(self.actor_attention.state_dict(),os.path.join(self.output_dir,f'actor_attention_{episode}.pth'))
        torch.save(self.actor_attention_target.state_dict(),os.path.join(self.output_dir,f'actor_attention_target_{episode}.pth'))

        torch.save(self.critic.state_dict(),os.path.join(self.output_dir,f'critic_{episode}.pth'))
        torch.save(self.critic_target.state_dict(),os.path.join(self.output_dir,f'critic_target_{episode}.pth'))

        torch.save(self.critic_attention.state_dict(),os.path.join(self.output_dir,f'critic_attention_{episode}.pth'))
        torch.save(self.critic_attention_target.state_dict(),os.path.join(self.output_dir,f'critic_attention_target_{episode}.pth'))

        with open(os.path.join(self.output_dir,'episode_cases.json'),'w') as f:
            json.dump(str(self.episode_cases_trackor),f)
        with open(os.path.join(self.output_dir,'episode_deaths.json'),'w') as f:
            json.dump(str(self.episode_deaths_trackor),f)
        with open(os.path.join(self.output_dir,'critic_loss.json'),'w') as f:
            json.dump(str(self.critic_loss_trackor),f)
        with open(os.path.join(self.output_dir,'actor_loss.json'),'w') as f:
            json.dump(str(self.actor_loss_trackor),f)
        with open(os.path.join(self.output_dir,'action_entropy.json'),'w') as f:
            json.dump(str(self.action_entropy_trackor),f)


if __name__ == '__main__':
    train_platform=MARL()
    # train_platform.test_simulation()
    # train_platform.test_network()
    train_platform.train()