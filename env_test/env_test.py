import gym
import gym_airsim_multirotor
from cv2 import getTickCount, getTickFrequency

env = gym.make("airsim-simple-dynamics-v0")
# env.read_config('gym_airsim_multirotor/envs/config.ini')

env.reset()

step = 0
loop_start = getTickCount()
for i in range (100):
    loop_start = getTickCount()
    
    action = [5, 0]    
    obs, reward, done, info = env.step(action)
    step += 1
    
    loop_time = getTickCount() - loop_start
    total_time=loop_time/(getTickFrequency())
    FPS=1/total_time
    print(FPS)

    if done:
        env.reset()
        print(info)
