# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:59:13 2025

@author: Francois le Roux
"""

from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot = False
plot_values = True
plot_square_error = False
zoom = False
radius_check = False
plot_cores = True

if plot or plot_values or plot_square_error or plot_cores:
    import matplotlib as mpl
    mpl.rcParams['font.family'] = "serif"
    mpl.rcParams['font.serif'] = "cmr10"
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['axes.formatter.use_mathtext'] = True

folder = 'Apr25'
A = pd.read_csv(f"{folder}/cell_centre_A.ascii")
B = pd.read_csv(f"{folder}/cell_centre_B.ascii")
C = pd.read_csv(f"{folder}/cell_centre_C.ascii")
D = pd.read_csv(f"{folder}/cell_centre_D.ascii")

# data_options = [A,B,C,D]
# data_options_name = ['A','B','C','D']

# data_options = [B,D]
# data_options_name = ['B', 'D']

data_options = [A,B,C,D]
data_options_name = ['A','B','C','D']

# data_options = [C]
# data_options_name = ['C']
####
experimental_x = np.array([0.66,0.97,1.42,1.88])
experimental_circulation = np.array([0.240714286,0.190952381,0.14047619,0.121666667])
####
vortex_cores = {}
####
if plot_values:
    fig_circ, ax_circ = plt.subplots()
for case_number in range(len(data_options)):
    data = data_options[case_number]
    name = data_options_name[case_number]
    ##
    data = data.rename(columns = lambda x: x.strip())
    # Building of KDTree
    kddata = np.zeros((len(data),3))
    kddata[:,0] = data['x-coordinate']
    kddata[:,1] = data['y-coordinate']
    kddata[:,2] = data['z-coordinate']
    #
    kdtree = KDTree(kddata)
    #
    x_list = [0.66,0.97,1.42,1.88] #Check to maybe extend this list
    # x_list = [0.565, 0.66, 0.77, 0.97, 1.12, 1.15, 1.27, 1.31, 1.42, 1.57, 1.73, 1.88]
    circulation = np.zeros((len(x_list)))
    ####
    if plot_square_error:
        fig_e, ax_e = plt.subplots()
        ax_e.set_title(f'Case {name}: Circulation Error for Different Radii',fontsize=20)
    ####
    #
    print(f"Case {name}")
    #
    # r = 14 # radius
    if radius_check:
        if name == 'A':
            r_start = 8
            r_finish = 20
        elif name == 'B':
            r_start = 9
            r_finish = 21
        elif name == 'C':
            r_start = 8
            r_finish = 19
        elif name == 'D':
            r_start = 9
            r_finish = 21
        else:
            r_start = 8
            r_finish = 20
    else:
        if name == 'A':
            if plot_values:  
                r_start = 11
                r_finish = 12
            else:
                r_start = 15
                r_finish = 16
        elif name == 'B':
            r_start = 16
            r_finish = 17
        elif name == 'C':
            r_start = 14
            r_finish = 15
        elif name == 'D':
            r_start = 19
            r_finish = 20
        else:
            r_start = 8
            r_finish = 9
    if len(x_list) == 4:
        squared_error = np.zeros((len(range(r_start,r_finish))))
        squared_error_counter = 0
    for radius in range(r_start,r_finish):
        r = radius
        print(f'Radius = {r}')
        circulation_counter = 0

        for i in range(len(x_list)):
            x = x_list[i]
            # Get a and b (locations of vortex center)
            data_x = data.loc[(round(data['x-coordinate'],3) == x)]
            iv_p = data_x['pressure'].idxmin()
            iv_h = data_x['helicity'].idxmin()
            iv_l2c = data_x['lambda2-criterion'].idxmin()
            iv_q = data_x['q-criterion'].idxmax()
            # All results same, therefore using pressure
            
            a = round(data_x['z-coordinate'].loc[iv_p]*1000,2) 
            b = round(data_x['y-coordinate'].loc[iv_p]*1000,2)
        
            if i == 0:
                vortex_cores[name] = {x: {"Z": a, "Y": b}}
            else:
                vortex_cores[name].update({x: {"Z": a, "Y": b}})
            
            # All values returned were roughly the same, so only use iv_p
            
            # Quiver plot
            Z = data['z-coordinate'].loc[(round(data['x-coordinate'],2) == x)]
            Y = data['y-coordinate'].loc[(round(data['x-coordinate'],2) == x)]
            V = data['y-velocity'].loc[(round(data['x-coordinate'],2) == x)]
            W = data['z-velocity'].loc[(round(data['x-coordinate'],2) == x)]
            
            if plot:        
                fig, ax = plt.subplots()
                ax.quiver(Z,Y,W,V)
                ax.plot(a/1000,b/1000,'o',label='Vortex Core')
                ax.set_aspect('equal')
                ax.set_title(f'Mesh {name}: Velocity Field at x = {x} m',fontsize=20)
                ax.set_ylabel('Y [m]', fontsize=18)
                ax.set_xlabel('Z [m]', fontsize=18)
            
            num_points = 40 #Check value
            results = np.zeros((num_points,6))
            counter = 0
            for theta in np.linspace(0,2*np.pi,num_points):
                y = b + r*np.sin(theta)
                z = a + r*np.cos(theta)
                #
                point = np.array([[x,y/1000,z/1000]])
                d,p = kdtree.query(point,k=1)
                #
                xval = kddata[p[0],0]
                yval = kddata[p[0],1]
                zval = kddata[p[0],2]
                #
                values_to_look_at = data.loc[(data['x-coordinate'] == xval) & (data['y-coordinate'] == yval) & (data['z-coordinate'] == zval)]
                #
                u = values_to_look_at['x-velocity'].values[0]
                v = values_to_look_at['y-velocity'].values[0]
                w = values_to_look_at['z-velocity'].values[0]
                #
                results[counter,0] = xval
                results[counter,1] = yval
                results[counter,2] = zval
                results[counter,3] = u
                results[counter,4] = v
                results[counter,5] = w
                counter += 1
                #
            if plot:
                ax.plot(results[:,2],results[:,1],'.',color='tab:red',label=f'Circulation for r={r}')
                ax.legend(fontsize=16)
                fig.savefig(f'mass_figs/{name}_{i}.pdf',pad_inches=0, bbox_inches='tight', format='pdf')
            #
            summation = 0
            for j in range(len(results)-1):
                dz = results[j+1,2] - results[j,2]
                dy = results[j+1,1] - results[j,1]
                vdy = results[j,4]*dy
                wdz = results[j,5]*dz
                
                summation = summation + vdy + wdz
            circulation[circulation_counter] = summation
            circulation_counter += 1
            
            print(f'Circulation at {x} = {summation}')
            ##
            # Squared Error Calculation
            if len(x_list) == 4:
                ans = experimental_circulation[i]
                error_squared = np.square(ans-summation) 
                squared_error[squared_error_counter] = squared_error[squared_error_counter] + error_squared
            ##
        
        ##
        if plot_values:
            ax_circ.plot(np.array(x_list),circulation,label=f'Case {name}, E={np.round(squared_error[squared_error_counter],6)}')
        if len(x_list) == 4:
            squared_error_counter += 1
    
        
    if plot_square_error:
        ax_e.plot(range(r_start,r_finish),squared_error,'o-',color='tab:green',label='Squared Error')
        ax_e.set_xlabel('Radius',fontsize=18)
        ax_e.set_ylabel('Squared Error',fontsize=18)
        #####
        circle_rad = 10
        if name == 'A':
            xmin = 15
            ymin = squared_error[7]
            pmin = [xmin,ymin]
            ax_e.plot(pmin[0],pmin[1],'o',markersize=circle_rad*2,mec='tab:red',mfc='none',mew=2,label='Minimum')
            #######
            if zoom:
                import matplotlib.patches as patches
                rect = patches.Rectangle((13.5,0.0005),2,0.0015,color='k',fill=False)
                ax_e.add_patch(rect)
                # ax_e.plot(13.9,0.00105,'s',markersize=12,mec='k',mfc='none',mew=2)
                #####
                ri_ax = fig_e.add_axes([0.4,0.5,0.4,0.2])
                ri_ax.plot(range(13,17),squared_error[5:9],'o-',color='tab:green')
                ri_ax.plot(pmin[0],pmin[1],'o',markersize=circle_rad*2,mec='tab:red',mfc='none',mew=2)
                ri_ax.set_xlim(13.9,15.1)
                ri_ax.set_ylim(0.00105,0.00109)
                ri_ax.set_xticks([14.0,15.0])
                #####
                #####
                ax_e.annotate("",
                              xy=(14.5,0.002), xytext=(0,120), textcoords='offset points',
                              color='k',size = 18, va='center',
                              bbox=dict(boxstyle='square,pad=0.3',fc='w',color='k'),
                              arrowprops=dict(
                                  arrowstyle="<|-",
                              color='k'))
        if name == 'B':
            xmin = 16
            ymin = squared_error[7]
            pmin = [xmin,ymin]
            ax_e.plot(pmin[0],pmin[1],'o',markersize=circle_rad*2,mec='tab:red',mfc='none',mew=2,label='Minimum')
        if name == 'C':
            xmin = 14
            ymin = squared_error[6]
            pmin = [xmin,ymin]
            ax_e.plot(pmin[0],pmin[1],'o',markersize=circle_rad*2,mec='tab:red',mfc='none',mew=2,label='Minimum')
        if name == 'D':
            xmin = 19
            ymin = squared_error[10]
            pmin = [xmin,ymin]
            ax_e.plot(pmin[0],pmin[1],'o',markersize=circle_rad*2,mec='tab:red',mfc='none',mew=2,label='Minimum')
        ax_e.legend(fontsize=16)
        
if plot_values:
    ax_circ.plot(experimental_x,experimental_circulation,'s',label='Experimental Data',color='tab:purple')
    ax_circ.errorbar(experimental_x, experimental_circulation, yerr=experimental_circulation*0.05,marker='s',color='tab:purple',linestyle="")
    ax_circ.legend(fontsize=16)
    ax_circ.set_xlabel('X [m]',fontsize=18)
    ax_circ.set_ylabel('Circulation [m$^2$/s]',fontsize=18)
    ax_circ.set_title('Comparison of Simulation and Experimental Circulation Data', fontsize=18)
    
Zexp_cores = np.array([3.125,5.143,6.828,7.711])/100
Yexp_cores = np.array([1.25,1.45,2,2.29])/100

Zpaper = np.array([3.276157804,5.29209622,6.775862069,7.74137931])/100
Z_thesis = np.array([3.125,5.143369176,6.828828829,7.711711712])/100

Ypaper = np.array([1.43231441,1.460869565,1.956331878,2.431034483])/100
Y_thesis = np.array([1.25,1.45,2,2.29])/100

Zmiddel = (Zpaper+Z_thesis)/2
Ymiddel = (Ypaper+Y_thesis)/2

Zp = np.array([3.276157804,5.29209622,6.775862069,7.74137931])/100
Zt = np.array([3.125,5.143369176,6.828828829,7.711711712])/100

Yp = np.array([1.43231441,1.460869565,1.956331878,2.431034483])/100
Yt = np.array([1.25,1.45,2,2.29])/100

Zm = (Zp+Zt)/2
Ym = (Yp+Yt)/2

if plot_cores:
    figc, axc = plt.subplots()
    for mesh in vortex_cores.keys():
        Zlocs = np.zeros(4)
        Ylocs = np.zeros(4)
        loccounter = 0
        for location in vortex_cores[mesh].keys():
            Z = vortex_cores[mesh][location]['Z']/1000
            Zlocs[loccounter] = Z
            Y = vortex_cores[mesh][location]['Y']/1000
            Ylocs[loccounter] = Y
            loccounter += 1
        axc.plot(Zlocs,Ylocs,'o-',label=mesh)
    # axc.plot(Zexp_cores,Yexp_cores,'s',label='Experimental')
    
    axc.plot(Zm,Ym,'s',color='k',label='Experimental')
    
    for i in range(len(Zp)):
        if i == 3:
            up = 1.06
            down = 0.94
        else:
            up = 1.05
            down = 0.95
        if Zp[i] >= Zt[i]:
            x = [up*Zp[i],down*Zt[i]]
        else:
            x = [down*Zp[i],up*Zt[i]]
        if Yp[i] > Yt[i]:
            y1 = [up*Yp[i],up*Yp[i]]
            y2 = [down*Yt[i],down*Yt[i]]
        else:
            y1 = [down*Yp[i],down*Yp[i]]
            y2 =[up*Yt[i],up*Yt[i]]
        
        axc.fill_between(x,y1,y2,color='tab:blue',alpha=0.4)
        axc.legend(fontsize=16)
        axc.set_title("Comparison of Vortex Core Locations",fontsize=18)
        axc.set_xlabel("Z [m]", fontsize=16)
        axc.set_ylabel("Y [m]", fontsize=16)
        axc.annotate("X = 0.66 m",
                     xy=[Zm[0],Ym[0]],xytext=(-30,80),textcoords='offset points',
                     color='k',size=16,va="center",
                     bbox=dict(
                         boxstyle="square,pad=0.2",fc="w",color='k'),
                     arrowprops=dict(
                         arrowstyle="-|>",connectionstyle="arc3",
                         color="k"))
        axc.annotate("X = 0.97 m",
                     xy=[Zm[1],Ym[1]],xytext=(-30,-60),textcoords='offset points',
                     color='k',size=16,va="center",
                     bbox=dict(
                         boxstyle="square,pad=0.2",fc="w",color='k'),
                     arrowprops=dict(
                         arrowstyle="-|>",connectionstyle="arc3",
                         color="k"))
        axc.annotate("X = 1.42 m",
                     xy=[Zm[2],Ym[2]],xytext=(-150,10),textcoords='offset points',
                     color='k',size=16,va="center",
                     bbox=dict(
                         boxstyle="square,pad=0.2",fc="w",color='k'),
                     arrowprops=dict(
                         arrowstyle="-|>",connectionstyle="arc3",
                         color="k"))
        axc.annotate("X = 1.88 m",
                     xy=[Zm[3],Ym[3]],xytext=(-150,10),textcoords='offset points',
                     color='k',size=16,va="center",
                     bbox=dict(
                         boxstyle="square,pad=0.2",fc="w",color='k'),
                     arrowprops=dict(
                         arrowstyle="-|>",connectionstyle="arc3",
                         color="k"))
        