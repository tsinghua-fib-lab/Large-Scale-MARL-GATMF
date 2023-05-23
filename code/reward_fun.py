import numpy as np

def reward_fun(s,s1,D_weight=0,option=0):
    reward_C=s1[:,:,1]-s[:,:,1]
    reward_D=s1[:,:,2]-s[:,:,2]

    if option==0:
        reward=-np.clip(np.sqrt(reward_C+D_weight*reward_D),0,0.4)[:,:,np.newaxis]

    elif option==1:
        reward=-1000*(reward_C+D_weight*reward_D)[:,:,np.newaxis]

    elif option==2:
        reward=-100*(reward_C+D_weight*reward_D)[:,:,np.newaxis]

    elif option==3:
        reward=-10*(reward_C+D_weight*reward_D)[:,:,np.newaxis]

    return reward