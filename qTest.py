#Isaiah English
#CSC 499
#Q-Learning

from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils
import numpy as np

#CITATION: sample_mission_loader file from Malmo 0.36.0
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

#starts timer for run
startTime = time.time()
j = 0
convergeTest = 0
changeTest = 0
y = 0
goalCheck = 0
runNum = 0
#counter in the act method
k = 0
oldRun = 0
alpha = 0.1
gamma = 0.1
epsilon = 1.0
maxEpsilon = 1.0
minEpsilon = 0.01
w = 0
p = 0
cond = True
totalTime = 0.0

class QAgent(object):
    
    i = 0

    #setup for the agent CITATION: tutorial_6 file from Malmo 0.36.0 
    def __init__(self, actions=[]):
        self.actions = actions
        self.qTable = {}
        
       
    #load model CITATION: sample_mission_loader file from Malmo 0.36.0
    def loadModel(self, model_file):
        with open(model_file) as f:
            self.qTable = json.load(f)
    
   
    #updated q values and sends the appropriate action to the agent
    def qlearn(self, world_state, agent_host, currentR):
        """take 1 action in response to the current world state"""
        #global p
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) #most recent observation
       
        
        currentS = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))

        #add to q table if not there
        if currentS not in self.qTable:
            self.qTable[currentS] = ([0] * len(self.actions))

        #update Q values
        if self.oldS is not None and self.oldA is not None:
            oldQ = self.qTable[self.oldS][self.oldA]
            self.qTable[self.oldS][self.oldA] = oldQ + alpha * (currentR + gamma * max(self.qTable[currentS]) - oldQ)


        #choose the next action.
        #print(epsilon)
        rnd = random.random()
        if rnd < epsilon:
            a = random.randint(0, len(self.actions) - 1)
        else:
            m = max(self.qTable[currentS])
            t = self.qTable[currentS].index(m)
            a = t
            
            
        #send the move to the agent
        agent_host.sendCommand(self.actions[a])
        self.oldS = currentS
        self.oldA = a

        return currentR
    
    w=0

    #tests for the agent solving the maze and grabs observation from the agent
    def run(self, agent_host):
       
        #print("start time: ", startTime)
        totalReward = 0
        currentR = 0
        tol = 0.01
        global runNum
        global j
        global convergeTest
        global changeTest
        global goalCheck
        global oldRun
        global cond
        global totalTime
        oldRun = runNum
        runNum = 0
        
        self.oldS = None
        self.oldA = None
        
        #wait for a valid observation, video frame observation stuff
        world_state = agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        #wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()
        world_state = agent_host.getWorldState()
        for err in world_state.errors:
            print(err)

        if not world_state.is_mission_running:
            return 0 #mission ended
            
        assert len(world_state.video_frames) > 0, 'No video frames!?'
        
        obs = json.loads( world_state.observations[-1].text )
        oldX = obs[u'XPos']
        oldZ = obs[u'ZPos']
        
        
        #take first move
        totalReward += self.qlearn(world_state,agent_host,currentR)
        
        require_move = True
        check_expected_position = True
        
        #main loop:
        while world_state.is_mission_running:
            #i = 0
            #wait for the position to have changed and a reward received
            
            while True:
                world_state = agent_host.peekWorldState()
                if not world_state.is_mission_running:
                   
                    #print()
                    #i+=1
                    break
                if len(world_state.rewards) > 0 and not all(e.text=='{}' for e in world_state.observations):
                    obs = json.loads( world_state.observations[-1].text )
                    currentX = obs[u'XPos']
                    currentZ = obs[u'ZPos']
                    if require_move:
                        if math.hypot( currentX - oldX, currentZ - oldZ ) > tol:
                            #print()
                            #i+=1
                            i =0
                            break
                    else:
                        #print()
                        i+=1
                        break
                    
            #wait for a frame to arrive after that
            num_frames_seen = world_state.number_of_video_frames_since_last_state
            while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
                world_state = agent_host.peekWorldState()
                
            num_frames_before_get = len(world_state.video_frames)
            
            world_state = agent_host.getWorldState()
            for err in world_state.errors:
                print(err)
            currentR = sum(r.getValue() for r in world_state.rewards)

            
            
            if world_state.is_mission_running:
                assert len(world_state.video_frames) > 0, 'No video frames'
                num_frames_after_get = len(world_state.video_frames)
                assert num_frames_after_get >= num_frames_before_get, 'Fewer frames after getWorldState'
                frame = world_state.video_frames[-1]
                obs = json.loads( world_state.observations[-1].text )
                currentX = obs[u'XPos']
                currentZ = obs[u'ZPos']
                
                runNum += 1
                
                #test to see if maze has been solved
                if (runNum == 12) and ((currentX == 1.5 and currentZ == 10.5) or (currentX == 2.5 and currentZ == 11.5)):
                    #print("Action number that gets us here:",runNum)
                    goalCheck += 1
                elif oldRun != 12:
                    goalCheck = 0

                #if reached goal 10 times in a row, we're done    
                if goalCheck == 10:
                    endTime = time.time()
                    totalTime = endTime - startTime
                    print("converged with a time of:", totalTime)
                    cond = False
                    
               
                oldX = currentX
                oldZ = currentZ
                # act
                totalReward += self.qlearn(world_state, agent_host, currentR)
                
        #process final reward
        
        totalReward += currentR

        #update Q values for terminal state
        if self.oldS is not None and self.oldA is not None:
            oldQ = self.qTable[self.oldS][self.oldA]
            self.qTable[self.oldS][self.oldA] = oldQ + alpha * ( currentR + gamma * max(self.qTable[self.oldS])- oldQ )
    
    
        return totalReward
    
  

agent_host = MalmoPython.AgentHost()


#args for agent CITATION: sample_mission_loader file from Malmo 0.36.0
mission_file = './environment.xml'
agent_host.addOptionalStringArgument('mission_file',
    'Path/to/file from which to load the mission.', mission_file)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
malmoutils.parse_command_line(agent_host)


tests = 15
for x in range(tests):
    #Agent setup citation: CITATION: tutorial_6 file from Malmo 0.36.0
    actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
    agent = QAgent(actions=actionSet)


    
    #set up for the mission CITATION: sample_mission_loader file from Malmo 0.36.0
    mission_file = './environment.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.removeAllCommandHandlers()
    my_mission.allowAllDiscreteMovementCommands()
    my_mission.requestVideo( 320, 240 ) #changed 320 to 900 (sizes for large moniter)
    my_mission.setViewpoint( 1 )
    
    #setup for client CITATION: sample_mission_loader file from Malmo 0.36.0
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) #add Minecraft machines here as available
    max_retries = 3
    agentID = 0
    expID = 'Qlearning'
    
    w = 0
    cond = True
    epsilon = 1.0
    startTime = time.time()
    goalCheck = 0
    #amount of times agent can go through maze
    runs = 500
    cumulativeRewards = []
    #f = open("runs.txt", "a")
    while w < 500 and cond:
        w+=1
        print("Trial:", x)
        print("Run %d:" % (  w ))
        epsilon = minEpsilon + (maxEpsilon - minEpsilon)*np.exp(-0.05*w)
        p += 1
        #f.write(str(w)+'\n')
    
        my_mission_record = malmoutils.get_default_recording_object(agent_host, "./save_%s-map%d-rep%d" % (expID, 1, w))
    
        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, w) )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2.5)
    
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print()

        #run agent
        cumulativeReward = agent.run(agent_host)
        #summate cumulative rewards
        cumulativeRewards += [ cumulativeReward ]

        #clean up
        time.sleep(0.5)
        
    #puts outputs to files  
    f = open("runs.txt", "a")
    g = open("time.txt", "a")
    f.write(str(w)+'\n')
    g.write(str(totalTime)+'\n')
    f.close()
    g.close()
    
print("WOrked?")
    
