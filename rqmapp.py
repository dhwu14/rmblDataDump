# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# import rqm
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as pp
import _pickle as pickle
import scipy.special

import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly import graph_objs as go 
from plotly.graph_objs import *
import plotly.tools as tls 
import plotly.express as px
import base64

N=12
b=6
W=1
Wx=0
J=1
U=1
mu=1
A=1
nk=3
v=0.1
a1=1
a2=1

data={}
# En={}
n=20
SquareLog=np.logspace(-5,5,n)[8:13]
Wlog=np.logspace(-5,5,int(2*n))[15:25]
numStates=int(scipy.special.binom(12,6))

for i in range(len(SquareLog)):
  for j in range(len(Wlog)):
    data[i,j]=[]
    # En[i,j]=[]

def EnergyGapStat(ev):
    dis   = 0
    # ev    = np.sort(np.real(ev),axis=None)
    diff  = []
    for i in range(len(ev)-1):
        diff.append(ev[i+1]-ev[i])
    for i in range(len(diff)-1):
        dis += np.min([diff[i],diff[i+1]])/np.max([diff[i],diff[i+1]])
    return dis/(np.double(len(ev))-1.0+1e-8)

# path="/Users/dhwu/Desktop/refael/program/cluster programs/data/data 02-16-2021/Round2/ROI/"
path = "round5/"

cluster = ["region3row8/","region3row9/","region4row8/","region4row9/",\
           "region1row10/","region1row11/","region1row12/",\
           "region2row10/","region2row11/","region2row12/"]

for folder in cluster:
  pwd = path+folder+"L12b6_W0Zoom"
    # chi[i,j]+=np.mean(np.real(np.array(temp_data[0])-np.array(temp_data[1]))[-1])/6
    # drag[i,j]+=np.mean(np.real((np.array(temp_data[0])+np.array(temp_data[1])))[-1])/2/6

  for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
      # print(pwd+str(1e4+i*100+j)+".dat")
      if (os.path.isfile(pwd+str(1e4+(i+8)*100+(j+15))+"_temp.dat")):
        # print(pwd+str(1e4+(i+5)*100+(j+15))+"_temp.dat"+" exists!")
        f=open(pwd+str(1e4+(i+8)*100+(j+15))+"_temp.dat", 'rb')
        temp_data=pickle.load(f)
        f.close()
        # print("Ary averaged: "+str(len(temp_data[0])))
        # print("System size: "+str(temp_data[1]['N'])+"; particles: "+str(temp_data[1]['b']))
        # print("match? ",temp_data[1]['A']==SquareLog[i],temp_data[1]['W']==SquareLog[j])

        # print(len(temp_data),len(temp_data[0]),len(temp_data[0][0]))
        # print(len(temp_data[0]))
        # print(temp_data[1]['v'])
        data[i,j].append(temp_data[0])
        # En[i,j].append(temp_data[2])

# print("diag12 files matched: "+str(countMac0)+"; iter2 files matched: "+str(countMac1))

# path2="/Users/dhwu/Desktop/refael/program/cluster programs/data/data 02-16-2021/Round2/ROIslow/"
path2="round5v2/"

data2={}
# En2={}
for i in range(len(SquareLog)):
  for j in range(len(Wlog)):
    data2[i,j]=[]
    # En2[i,j]=[]

for folder in cluster:
  pwd2=path2+folder+"L12b6_W0Zoom"
  for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        # print(pwd+str(1e4+i*100+j)+".dat")
      if (os.path.isfile(pwd2+str(1e4+(i+8)*100+(j+15))+"_temp.dat")):
        # print(pwd2+str(1e4+(i+5)*100+(j+15))+"_temp.dat"+" exists!")
        f=open(pwd2+str(1e4+(i+8)*100+(j+15))+"_temp.dat", 'rb')
        temp_data=pickle.load(f)
        f.close()
        # print("Ary averaged: "+str(len(temp_data[0])))
        # print("System size: "+str(temp_data[1]['N'])+"; particles: "+str(temp_data[1]['b']))
        # print("match? ",temp_data[1]['A']==SquareLog[i],temp_data[1]['W']==SquareLog[j])
        # print(temp_data[1]['v'])
        data[i,j].append(temp_data[0])
        # En2[i,j].append(temp_data[2])

# path3="/Users/dhwu/Desktop/refael/program/cluster programs/data/data 02-16-2021/Round2/ROIfast/"
path3="round5v3/"

data3={}
# En3={}
for i in range(len(SquareLog)):
  for j in range(len(Wlog)):
    data3[i,j]=[]
    # En3[i,j]=[]

for folder in cluster:
  pwd3=path3+folder+"L12b6_W0Zoom"
  for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        # print(pwd+str(1e4+i*100+j)+".dat")
      if (os.path.isfile(pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat")):
        # print(i,j,pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat"+" exists!")
        f=open(pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat", 'rb')
        temp_data=pickle.load(f)
        f.close()
        # print("Ary averaged: "+str(len(temp_data[0])))
        # print("System size: "+str(temp_data[1]['N'])+"; particles: "+str(temp_data[1]['b']))
        # print("match? ",temp_data[1]['A']==SquareLog[i],temp_data[1]['W']==SquareLog[j])
        # print(temp_data[1]['v'])
        data[i,j].append(temp_data[0])  # chooses the data; temp_data[1] is the dictionary of parameters
        # En3[i,j].append(temp_data[2])

path4="round5v4/"

data4={}
# En3={}
for i in range(len(SquareLog)):
  for j in range(len(Wlog)):
    data3[i,j]=[]
    # En3[i,j]=[]

for folder in cluster:
  pwd3=path3+folder+"L12b6_W0Zoom"
  for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        # print(pwd+str(1e4+i*100+j)+".dat")
      if (os.path.isfile(pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat")):
        # print(i,j,pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat"+" exists!")
        f=open(pwd3+str(1e4+(i+8)*100+(j+15))+"_temp.dat", 'rb')
        temp_data=pickle.load(f)
        f.close()
        # print("Ary averaged: "+str(len(temp_data[0])))
        # print("System size: "+str(temp_data[1]['N'])+"; particles: "+str(temp_data[1]['b']))
        # print("match? ",temp_data[1]['A']==SquareLog[i],temp_data[1]['W']==SquareLog[j])
        # print(temp_data[1]['v'])
        data[i,j].append(temp_data[0])  # chooses the data; temp_data[1] is the dictionary of parameters
        # En3[i,j].append(temp_data[2])


# print(len(data[0,0][0][0][0]))
count=np.zeros([len(SquareLog),len(Wlog)])
val={}
EnStat={}
for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        val[i,j]=[np.zeros([len(data[i,j][0][0][0]),len(data[i,j][0][0][0][0])],dtype=complex),\
                    np.zeros([len(data[i,j][0][0][1]),len(data[i,j][0][0][1][0])],dtype=complex),\
                    np.zeros(numStates,dtype=complex),np.zeros(numStates,dtype=complex)]
        EnStat[i,j]=0


for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        for k in range(len(data[i,j])): # len(data[i,j])=3 from the number of folders, i.e. v1-v3
            for l in range(len(data[i,j][k])): # iterate through the number of runs for each data point within a folder
                for m in range(len(data[i,j][k][l])): # the four results: +v, -v, E_{+v}, E_{-v}
                    # print(len(data[i,j][k][l][m]),len(data[i,j][k][l][m][0]))
                    val[i,j][m]+=data[i,j][k][l][m] # note that m=0 structure is 301 by 924 where the latter is storing the 924 states
                rind=EnergyGapStat(data[i,j][k][l][2])
                # print(i,j,k,l,rind)
                EnStat[i,j]+=rind
                count[i,j]+=1
        val[i,j]/=count[i,j]
        EnStat[i,j]/=count[i,j]
# print(data[0,0][0][0][2],EnergyGapStat(data[0,0][0][0][2]))

analysis2=[val,val,val]
tabData={}
for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        tabData[i,j]=[]

for i in range(len(SquareLog)):
    for j in range(len(Wlog)):
        for k in range(len(data[i,j])):
            for l in range(len(data[i,j][k])):
                tabData[i,j].append(data[i,j][k][l])

analysis=[tabData,tabData,tabData]
vChoice=[0.1,0.1,0.1]
# params=['W','Wx','n','b','J','U','mu','t','A','p','v','a1','a2','pot']
# rqm.CompState("States.dat")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

server=app.server

app.layout = html.Div([

    html.H1("Ratchet MBL drag",style={'text-align':'center'}),

    html.H3("Energy density choice", style={'text-align':'center', 'width':"40%"}),
    
    html.Div([
            
            # html.Label('Lattice sites n'),
            # dcc.Slider(id='n',
            #     min=2,
            #     max=12,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(2, 12+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=5,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Number of particles'),
            # dcc.Slider(id='b',
            #     min=0,
            #     max=12,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 12+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=2,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Hopping Amplitude J'),
            # dcc.Slider(id='J',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=1,
            #     #style={'width':"40%"}
            # ),

            # html.Label('On-site interaction strength U'),
            # dcc.Slider(id='U',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=1,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Chemical potential'),
            # dcc.Slider(id='mu',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=1,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Disorder parameter W'),
            # dcc.Slider(id='W',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i*10) if i == 1 else str(i*10) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=1,
            #     #style={'width':"40%"}
            # )

            html.Label('Energy density location'),
            dcc.Slider(id='enStart',
                min=0,
                max=numStates-1,
                marks={100*i: '{}'.format(100*i) if i == 0 else str(100*i) for i in range(0, 9)},
                #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
                value=int(numStates/2),
                #style={'width':"40%"}
            ),

            html.Label('Energy density window size'),
            dcc.Slider(id='winSize',
                min=1,
                max=numStates,
                marks={100*i: '{}'.format(100*i) if i == 0 else str(100*i) for i in range(0, 9)},
                #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
                value=20,
                #style={'width':"40%"}
            ),

            html.Label('Disorder realization no.'),
            dcc.Slider(id='disNum',
                min=1,
                max=int(np.min(count)),
                marks={i: '{}'.format(i) if i==0 else str(i) for i in range(1,int(np.min(count)))},
                value=1,
            )

        ], style={'columnCount': 3}),
    
    # html.H3("Potential parameters:", style={'text-align':'center', 'width':"40%"}),

    # html.Div([

            
            
            # html.Label('Potential type'),
            # dcc.Dropdown(id="pot",
            #     options=[
            #         {'label': u'Sawtooth potential', 'value': 'sw'},
            #         {'label': 'Reverse sawtooth potential', 'value': 'rsw'},
            #         {'label': 'Bichromatic flashing potential', 'value': 'bi'}
            #     ],
            #     multi=False,
            #     value='sw',
            #     style={'width':"70%"}
            # ),

            # html.Div(id='output_container',children=[]),

            # html.Label('Potential amplitude A'),
            # dcc.Slider(id='A',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i*10) if i == 1 else str(i*10) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=3,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Period p'),
            # dcc.Slider(id='p',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=2,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Velocity v'),
            # dcc.Slider(id='v',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i*0.1) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=1,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Time depndent potential amplitude a2'),
            # dcc.Slider(id='a2',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=5,
            #     #style={'width':"40%"}
            # ),

            # html.Label('Quasiperiodic parameter a1'),
            # dcc.Slider(id='a1',
            #     min=0,
            #     max=9,
            #     marks={i: '{}'.format(i) if i == 1 else str(i) for i in range(0, 9+1)},
            #     #marks={i: 'J {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
            #     value=5,
            #     #style={'width':"40%"}
            # )
        # ], style={'columnCount': 1}),

    html.Div([
        html.Label('energy density location:'),
        html.Div(id='updatemode-output-container',children=[]),

        html.Label('window size:'),
        html.Div(id='updatemode-output-container2',children=[]),
    ], style={'margin-top': 20, 'columnCount': 2}),

    html.Div([
        html.Label('Velocity'),
        dcc.Dropdown(id="v",
                options=[
                    {'label': '0.01', 'value': 'slow'},
                    {'label': '0.1', 'value': 'normal'},
                    {'label': '1', 'value': 'fast'}
                ],
                multi=False,
                value='normal',
                style={'width':"70%"}
            ),

        html.Label('Average disorder?'),
        dcc.Dropdown(id="Wavg",
                options=[
                    {'label': 'yes', 'value': 'yes'},
                    {'label': 'no', 'value': 'no'}
                ],
                multi=False,
                value='no',
                style={'width':"70%"}
            )
    ], style={'columnCount': 2}),

    

    # html.Div([
    #     html.Label('Evolution time'),
    #     dcc.Input(id='t',value=100, type='number'),

    #     html.Label('Disorder parameter: '),
    #     html.Div(id='output_container2',children=[]),
    # ], style={'columnCount': 2}),
    

    html.Br(),

    

    html.Div([
    	html.H2("energy gap statistics:",style={'text-align':'center', 'width':"40%"}),
        dcc.Graph(id='r-index',figure={}),
        html.H2("heatmap of rectification in parameter space:", style={'text-align':'center', 'width':"40%"}),
        dcc.Graph(id='rec',figure={}),
        html.H2("heatmap of mobility in parameter space:", style={'text-align':'center', 'width':"40%"}),
        dcc.Graph(id='mob',figure={}),
        
        # html.H2("v=-|v|:", style={'text-align':'center', 'width':"40%"}),
        # dcc.Graph(id='results2',figure={}),
        # html.H2(" ", style={'text-align':'center', 'width':"40%"}),
        # dcc.Graph(id='results2Histo',figure={})
    ], style={'columnCount': 3}),

    html.H2("Specific instances", style={'text-align':'center', 'width':"40%"}),

    html.Div([
        html.Label("Amplitude of static ratchet"),
        dcc.Input(id='A',min=0,max=4,value=3,type='number'),

        html.Label("Disorder amplitude"),
        dcc.Input(id='W',min=0,max=9,value=7,type='number'),
    ]),

    html.Div([
        dcc.Graph(id='recIns',figure={}),
        dcc.Graph(id='mobIns',figure={})
        ], style={'columnCount': 2})
    

])#, style={'columnCount': 2})

# -------------------------------------------------------------------------------------------------
# Connect the plotly graphs with Dash components

@app.callback(
    [Output(component_id='r-index',component_property='figure'),
    Output(component_id='rec',component_property='figure'),
    Output(component_id='mob',component_property='figure'),
    Output(component_id='recIns',component_property='figure'),
    Output(component_id='mobIns',component_property='figure'),
    Output(component_id='updatemode-output-container',component_property='children'),
    Output(component_id='updatemode-output-container2',component_property='children')],
    [Input(component_id='enStart',component_property='value'),
    Input(component_id='winSize',component_property='value'),
    Input(component_id='disNum',component_property='value'),
    Input(component_id='v',component_property='value'),
    Input(component_id='Wavg',component_property='value'),
    Input(component_id='A',component_property='value'),
    Input(component_id='W',component_property='value')
    # Output(component_id='output_container',component_property='children'),
    # Output(component_id='output_container2',component_property='children'),
    # Output(component_id='results1', component_property='figure'),
    # Output(component_id='results1Histo', component_property='figure'),
    # Output(component_id='results2', component_property='figure'),
    # Output(component_id='results2Histo', component_property='figure'),
    # Output(component_id='results3', component_property='figure'),
    # Output(component_id='results3Histo', component_property='figure')],
    # [Input(component_id='pot', component_property='value'),
    # Input(component_id='n', component_property='value'),
    # Input(component_id='b', component_property='value'),
    # Input(component_id='J', component_property='value'),
    # Input(component_id='U', component_property='value'),
    # Input(component_id='mu', component_property='value'),
    # Input(component_id='W', component_property='value'),
    # Input(component_id='t', component_property='value'),
    # Input(component_id='A', component_property='value'),
    # Input(component_id='v', component_property='value'),
    # Input(component_id='p', component_property='value'),
    # Input(component_id='a1', component_property='value'),
    # Input(component_id='a2', component_property='value')
    ]
)

def update_graph(enStartSelect, winSizeSelect, disNumSelect, vSelect, WavgSelect, ASelect, WSelect):
    # global Runs,output
    # exists=False
    # rev=False
    # val=0
    #print(type(potSelect),potSelect)
    # print(disNumSelect)
    # print(WavgSelect)
    global analysis, vChoice, SquareLog, Wlog, analysis2, EnStat
    if vSelect=='slow':
        choice=0
    elif vSelect=='normal':
        choice=1
    elif vSelect=='fast':
        choice=2

    # print(choice)

    Analysis=analysis[choice]
    if WavgSelect=='yes':
        # FIXIT
        Analysis=analysis2[choice]
    elif WavgSelect=='no':
        # count=1
        Analysis=analysis[choice]

    A=SquareLog[ASelect]
    W=Wlog[WSelect]

    begin=enStartSelect
    size=winSizeSelect

    # print(begin)
    # print(size)

    chi=np.zeros([len(SquareLog),len(Wlog)],dtype=complex)    # susceptibility
    drag=np.zeros([len(SquareLog),len(Wlog)],dtype=complex) # ratchetness
    var=np.zeros([len(SquareLog),len(Wlog)],dtype=complex)
    chiAbs=np.zeros([len(SquareLog),len(Wlog)],dtype=complex)   # susceptibility
    dragAbs=np.zeros([len(SquareLog),len(Wlog)],dtype=complex)  # ratchetness
    varAbs=np.zeros([len(SquareLog),len(Wlog)],dtype=complex)
    mask=np.ones([len(SquareLog),len(Wlog)],dtype=bool)
    rindex=np.zeros([len(SquareLog),len(Wlog)])

    # print(len(Analysis[len(SquareLog)-1,len(Wlog)-1]))
    print(WavgSelect)

    for i in range(len(SquareLog)):
        for j in range(len(Wlog)):
          # if(Analysis[i,j]==[]):
          #   chi[i,j]=0
          #   drag[i,j]=0
          # else:
          if True:
            mask[i,j]=False
            # count=0
            # for k in range(len(Analysis[i,j])):
            #   # print(i,j,len(data[i,j]),len(data[i,j][k]))
            #   for l in range(len(Analysis[i,j][k])):
            temp_data=analysis[0][i,j][0]
            if WavgSelect=='yes':
                Analysis=analysis2[choice]
                temp_data=Analysis[i,j]
            elif WavgSelect=='no':
                # count=1
                Analysis=analysis[choice]
                temp_data=Analysis[i,j][disNumSelect-1]
            chiAbs[i,j]+=np.mean(np.abs((np.array(temp_data[0])-np.array(temp_data[1])))[-1])/2/6
            dragAbs[i,j]+=np.mean(np.abs((np.array(temp_data[0])+np.array(temp_data[1])))[-1])/2/6
                # varAbs[i,j]+=np.mean(np.abs((np.array(temp_data[0])+np.array(temp_data[1]))/(np.array(temp_data[0])+np.array(temp_data[1])))[-1])
            chi[i,j]+=np.mean(np.real(np.array(temp_data[0])-np.array(temp_data[1]))[-1][begin:begin+size])/2/6
            drag[i,j]+=np.mean(np.real(np.array(temp_data[0])+np.array(temp_data[1]))[-1][begin:begin+size])/2/6
                # var[i,j]+=np.mean(np.real((np.array(temp_data[0])-np.array(temp_data[1]))/(np.array(temp_data[0])+np.array(temp_data[1])))[-1][numStates:numStates+20])
                # count += 1
            # print(i,j,count)

            chi[i,j]/=1.0
            drag[i,j]/=1.0
            # var[i,j]/=1.0*len(data[i,j])
            var[i,j]+=drag[i,j]/chi[i,j]
            chiAbs[i,j]/=1.0
            dragAbs[i,j]/=1.0
            # varAbs[i,j]/=1.0*len(data[i,j])
            varAbs[i,j]+=dragAbs[i,j]/chiAbs[i,j]
            rindex[i,j]=EnStat[i,j]

    yAxisNp=np.linspace(-5,5,n)[8:13]
    xAxisNp=np.linspace(-5,5,int(2*n))[15:25]
    yAxis=np.char.mod('%.3f',yAxisNp)
    xAxis=np.char.mod('%.3f',xAxisNp)

    rangeChi=np.max([np.abs(np.max(np.real(chi))),np.abs(np.min(np.real(chi)))])
    rangeDrag=np.max([np.abs(np.max(np.real(drag))),np.abs(np.min(np.real(drag)))])

    # print(EnStat)
    figRindex=px.imshow(np.flip(np.real(rindex),0)[:,:], 
        labels=dict(x="log(W/J)",y="log(A/a_2)"),x=xAxis,y=np.flip(yAxis,0),
        range_color=[0.39,0.53],color_continuous_scale=px.colors.diverging.RdBu)
    figRindex['layout']['xaxis']['type']='category'
    figRindex['layout']['yaxis']['type']='category'
    figChiHeatmap=px.imshow(np.flip(np.real(chi),0)[:,:], 
        labels=dict(x="log(W/J)",y="log(A/a_2)"),x=xAxis,y=np.flip(yAxis,0),
        range_color=[-rangeChi,rangeChi],color_continuous_scale=px.colors.diverging.RdBu)
    figChiHeatmap['layout']['xaxis']['type']='category'
    figChiHeatmap['layout']['yaxis']['type']='category'
    figDragHeatmap=px.imshow(np.flip(np.real(drag),0)[:,:],
        labels=dict(x="log(W/J)",y="log(A/a_2)"),x=xAxis,y=np.flip(yAxis,0),
        range_color=[-rangeDrag,rangeDrag],color_continuous_scale=px.colors.diverging.RdBu)
    figDragHeatmap['layout']['xaxis']['type']='category'
    figDragHeatmap['layout']['yaxis']['type']='category'

    # pots=['Sawtooth','Reverse sawtooth','Bichromatic flashing']
    # container="The chosen potential is {}".format(pots[val])
    # currentRun={'W':W,'Wx':Wx,'n':n,'b':b,'J':J,'U':U,'mu':mu,'t':t,'A':A,'p':p,'v':v,'a1':a1,'a2':a2,'pot':potSelect}
    # fText='L%.0f,'%n+'p%.0f,'%b+'J%.1f,'%J+'U%.1f,'%U+'mu%.1f,'%mu+'A%.1f,'%A+'omega%.1f,'%omega+'k%.1f,'%k+',a_1%.1f,'%a1+'a_2%.1f,'%a2+'W%.1f,'%W+'Wx%.1f'%Wx
    #print(fText)

    Ai=ASelect
    Wj=WSelect

    SingleData=analysis[choice][Ai,Wj][disNumSelect-1]
    energy=analysis[choice][Ai,Wj][disNumSelect-1][2]

    if WavgSelect=='yes':
        # FIXIT
        SingleData=analysis2[choice][Ai,Wj]
        energy=analysis2[choice][Ai,Wj][2]
    elif WavgSelect=='no':
        # count=1
        SingleData=analysis[choice][Ai,Wj][disNumSelect-1]
        energy=analysis[choice][Ai,Wj][disNumSelect-1][2]
    w=size
    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         if (i==0) and (j==0):
    #             continue
    #         val += data[i][j]
    #         count+=1

    # no disorder averaging
    outA=np.array(SingleData)
    # outA=np.array(val)#/(1.0*count)

    res=np.real((np.array(outA[0])+np.array(outA[1])))/2/6/(1.0)
    res2=np.real((np.array(outA[0])-np.array(outA[1])))/2/6/(1.0)

    ratchet=[]
    ratchetstd=[]
    drag=[]
    dragstd=[]
      # ratio=[]
    loc=[]
    locstd=[]
    iterations = int((numStates-np.mod(numStates,w))/w)
      # print(numStates,iterations)
      # fig =  pp.figure(fignum, figsize=(10/1.25*4/6*3,12/1.25/2),dpi=100,facecolor='w')
      # print(iterations)
    for ind in range(iterations):
        i = w*ind
        if ind==iterations-1:
            width=np.mod(numStates,w)
            ratchet.append(np.mean(np.real(res[-1][i:i+width])))
            drag.append(np.mean(np.real(res2[-1][i:i+width])))
            ratchetstd.append(np.std(np.real(res[-1][i:i+width])))
            dragstd.append(np.std(np.real(res2[-1][i:i+width])))
            loc.append(np.mean(np.real(energy[i:i+width])))
            locstd.append(np.std(np.real(energy[i:i+width])))
            break
        ratchet.append(np.mean(np.real(res[-1][i:i+w])))
        drag.append(np.mean(np.real(res2[-1][i:i+w])))
        ratchetstd.append(np.std(np.real(res[-1][i:i+w])))
        dragstd.append(np.std(np.real(res2[-1][i:i+w])))
        loc.append(np.mean(np.real(energy[i:i+w])))
        locstd.append(np.std(np.real(energy[i:i+w])))

    dfigPartChunkDrag={}
    dfigPartChunkChi={}
    dfigPartChunkDrag['x']=loc
    dfigPartChunkDrag['y']=ratchet
    dfigPartChunkDrag['err_x']=locstd
    dfigPartChunkDrag['err_y']=ratchetstd
    dfigPartChunkChi['x']=loc
    dfigPartChunkChi['y']=drag
    dfigPartChunkChi['err_x']=locstd
    dfigPartChunkChi['err_y']=dragstd

    figDrag=px.scatter(dfigPartChunkDrag,x='x',y='y',error_x='err_x',error_y='err_y',
                        labels={'x':'energy', 'y':'rectification'})
    figChi=px.scatter(dfigPartChunkChi,x='x',y='y',error_x='err_x',error_y='err_y',
                        labels={'x':'energy', 'y':'mobility'})

    # index=-1
    # data=[]
    # #print(len(Runs))
    # for i in range(len(Runs)):
    #     exists = True
    #     for key in params:
    #         if Runs[i][key]==currentRun[key]:
    #             exists = np.bool(exists*True)
    #         elif not(Runs[i][key]==currentRun[key]):
    #             exists = False
    #             break
    #     if exists:
    #         index = i
    #         break
    #print(exists)

    # if exists:
    #     if index==-1:
    #         print('Error: cannot find previous index')
    #         return
    #     data=output[index]

    # else:
    #     #print('here')
    #     output.append(rqm.Evo(rqm.Hamil,pot,n,b,J,U,mu,W,Wx,False,False,0.01,t,0.1,A,p,v,a1,a2,False,rev,fText))
    #     data=output[len(output)-1]
    #     Runs.append(currentRun)

    #fig=tls.mpl_to_plotly(rqm.graphWeb(data))
    # container2="v=+|v|: %.1f"%data[2][0]+", v=-|v|: %.1f"%data[3][0]
    
    # sample0=[]
    # sample1=[]
    # sample2=[]
    # for j in range(len(data[4])):    
    #     sample0.append([data[0][i][j] for i in range(len(data[0]))])
    #     sample1.append([data[1][i][j] for i in range(len(data[1]))])
    #     sample2.append([data[0][i][j]+data[1][i][j] for i in range(len(data[0]))])
    # #print(t,len(np.arange(0,t+1,1)),len(np.real(sample0[0]).tolist()))

    # h1d=np.real(data[0][-1])
    # h2d=np.real(data[1][-1])
    # h3d=np.real(data[0][-1]+data[1][-1])
    # #h1=np.histogram(h1d,bins=range(np.int(np.floor(np.min(h1d))),np.int(np.ceil(np.max(h1d))),25))
    # #h2=np.histogram(h2d,bins=range(np.int(np.floor(np.min(h2d))),np.int(np.ceil(np.max(h2d))),25))
    # h1=np.histogram(h1d,bins=25,range=(np.min(h1d),np.max(h1d)))
    # h2=np.histogram(h2d,bins=25,range=(np.min(h2d),np.max(h2d)))
    # h3=np.histogram(h3d,bins=25,range=(np.min(h3d),np.max(h3d)))
    # #bins1 = 0.5 * (h1[1][:-1] + h1[1][1:])
    # #bins2 = 0.5 * (h2[1][:-1] + h2[1][1:])

    # fig1  = px.line(x=np.arange(0,t+1,1), y=[np.real(sample0[i]).tolist() for i in range(len(data[4]))], labels={'x':'t','y':'current'})
    # fig2  = px.line(x=np.arange(0,t+1,1), y=[np.real(sample1[i]).tolist() for i in range(len(data[4]))], labels={'x':'t','y':'current'})
    # fig3  = px.line(x=np.arange(0,t+1,1), y=[np.real(sample2[i]).tolist() for i in range(len(data[4]))], labels={'x':'t','y':'current'})
    # fig1.update_layout(showlegend=False)
    # fig2.update_layout(showlegend=False)
    # fig3.update_layout(showlegend=False)

    # hist1 = px.bar(x=h1[1][:25].tolist(),y=h1[0],labels={'x':'current','y':'number of states'})
    # hist2 = px.bar(x=h2[1][:25].tolist(),y=h2[0],labels={'x':'current','y':'number of states'})
    # hist3 = px.bar(x=h3[1][:25].tolist(),y=h3[0],labels={'x':'current','y':'number of states'})



    return figRindex,figDragHeatmap,figChiHeatmap,figDrag,figChi,begin,size



if __name__ == '__main__':
    app.run_server(debug=True)