# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:12:43 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import itertools
import math
import imageio
from sklearn.cluster import KMeans
from tqdm import tqdm

def creat_line(a,b,noise,num_points,scale):
    ''' creat line in 3d: lin3D=a+b*scalar+noise, 
    in which l is 3d coordinates of the points on the line the shape is num_points*3
    a; a point on the line with shape 1*3
    b; the direction vector with shape 1*3
    noise: num_points*3
    ''' 
    scalar=scale*np.arange(0,1,1/num_points).reshape(-1,1)
    line3D=(a+np.dot(scalar,b)+noise);
    return line3D

def draw_lines(line3,view3d):
        fig = plt.figure()
        fig.set_size_inches(10.5, 10.5)

        ax = Axes3D(fig)
        plt.xlabel('x')
        plt.ylabel('y')
     
        ax.scatter3D(line3[:,0], line3[:,1], line3[:,2],marker='1', s=14,c='b')

        fig.savefig('test.png', dpi=50)
        if view3d:
            for angle in range(0,360,8):
                ax.view_init(elev=10., azim=angle)
                plt.savefig("movie%d.png" % angle)   
            outfilename = "my.gif"           
            frames = []
            for i in range(0,360,8):              
                im = imageio.imread('movie'+str(i)+'.png')            

                frames.append(im)                      
            imageio.mimsave(outfilename, frames, 'GIF', duration=0.1) 

        else:
            
            plt.show()

def creat_toy_data(num_lines,num_points,draw,noise):
    a_list=[]
    b_list=[]
    
    line3=[]
    for i in range(num_lines):
        a=np.random.rand(1,3)*100
        a_list.append(a)
        b=np.random.rand(1,3)
        b_list.append(b)
        l=creat_line(a,b,noise,num_points,50)
        line3.append(l)
    line3=np.array(line3).reshape(num_points*num_lines,3)
    if draw:
        draw_lines(line3,0)
    return line3,a_list,b_list

def sph2cart(az,el):
    '''Transform spherical coordinates to Cartesian'''
    cos_el=np.cos(el).reshape(-1,1)
    cos_az=np.cos(az).reshape(-1,1)
    sin_az=np.sin(az).reshape(-1,1)
    sin_el=np.sin(el).reshape(-1,1)
    cart=np.concatenate((cos_el*cos_az,cos_el*sin_az,sin_el),1)
    return cart

    
def shrink(p_prime, point_distance_threshold,num_point_threshold,merge_angle_point,cart):
    '''For the given set of points, if one point has less than 'num_point_threshold' number of points that are close to it, this point will be deleted 
    p_prime: given points
    point_distance_threshold: if the distance between two points is less than the threshold, that means two points are not close to each other
    merge_angle_point: is the function be used for merge both point and angles'''

    p_prime_list=[]
  
    sub_p_prime=p_prime[:,0].reshape(-1,1)
    co0=(sub_p_prime-sub_p_prime.T)**2
    
    sub_p_prime=p_prime[:,1].reshape(-1,1)
    co1=(sub_p_prime-sub_p_prime.T)**2
    
    sub_p_prime=p_prime[:,2].reshape(-1,1)
    co2=(sub_p_prime-sub_p_prime.T)**2
    
    co=np.sqrt(co0+co1+co2)<point_distance_threshold
    co_sum=np.sum(co,1)
    
        
    
    if merge_angle_point:
        return co[np.argmax(co_sum)]
    
    p_prime_copy=p_prime[co[np.argmax(co_sum)]]
   
    if p_prime_copy.shape[0]<num_point_threshold:
        return 0
    else:
        if len(cart)==1:
            co_index=np.where(co_sum>num_point_threshold)[0]
        else:
            co_index=np.where(co_sum==np.max(co_sum))[0]
            
        num_prime=co_index.size
     
        if num_prime==1:
            return p_prime_copy,co[np.argmax(co_sum)],num_prime
        else:
            for i in range(num_prime):
                p_prime_list.append(p_prime[co[co_index[i]]])
            return p_prime_list,co[np.argmax(co_sum)],num_prime



def hough3(p,cart,num_point_threshold,point_distance_threshold,merge_angle_point):
    '''for a point in p, every possible line passing through this point is sampled
    each line correspond to a nearest point to the origin and a set of azimuth and elevation angles in the parameter space
    Input:
    p: given points
    cart: all combination of azimuth and elevation
    
    Output:
    acc: 3D coordinates of all nearest points to the origin for all  sampled lines
    angle_index_list: each point in acc correspond to an angle index wich directs to a set of angles in cart
    
    '''

    n=0
    angle_index_list=[]
    acc=[]
    with tqdm(total=cart.shape[0], position=0, leave=True) as pbar:
        for i in tqdm(range(cart.shape[0]), leave=True, position=0):
            pbar.update()
            v=cart[i]
            B=p+v #%B is a point on the line
            origin=[0,0,0]
            k=(np.dot((origin-p),v))/(np.sum(np.absolute(B-p)**2,1)) #numOfPoints*1
            k=k.reshape(-1,1)
            p_prime=k*(B-p)+p #find the perpendicular foot coordinate
            
            
            if len(cart)==1:
                p=np.array(shrink(p_prime, point_distance_threshold,num_point_threshold,merge_angle_point,cart)[0])
                center_point=np.array([np.average(x,0) for x in p ])
                centers=np.concatenate((center_point,np.repeat(cart,center_point.shape[0],0)),1)
                return centers
            else:
               

                p_prime=shrink(p_prime,point_distance_threshold,num_point_threshold,merge_angle_point,cart)
                if p_prime==0:

                    continue
                else:
                    print(len(p_prime[0]))
                    n=n+p_prime[2]
                    if p_prime[2]==1:
                        p_prime=p_prime[0]
                        angle_index_list.append(i)
                        acc.append(p_prime)
                    else:
                        p_prime=p_prime[0]
                        for j in p_prime:
                            angle_index_list.append(i)
                            acc.append(j) 
    pbar.close()
    acc=np.array(acc)
    center_point=np.array([np.average(x,0) for x in acc])
    center_angles=cart[angle_index_list]
    centers=np.concatenate((center_point,center_angles),0)
    center_point=np.array([np.average(x,0) for x in acc])
    center_angles=cart[angle_index_list]
    centers=np.concatenate((center_point,center_angles),1)


    return centers
#%%
def cluster_points(cen, line3D):
    distance=[]
    Q1S=[]
    Q2S=[]
    for i in range(len(cen)):
        Q1=cen[i,0:3].reshape(1,-1)#1*3
        Q2=(centers[i,0:3]+cen[i,3:6]*10000).reshape(1,-1)#1*3
        Q1S.append(Q1)
        Q2S.append(Q2)
        d=np.linalg.norm(np.cross(Q2-Q1,line3D-Q1),axis=1)/np.linalg.norm(Q2-Q1)
        distance.append(d)
    distance=np.array(distance).T
    Q1S=np.squeeze(np.array(Q1S))
    Q2S=np.squeeze(np.array(Q2S))
    index=np.argmin(distance,1)   
    return index,Q1S,Q2S
def make_image(line3,k_centers,point_index,make_gif,name):
    fig = plt.figure()
    fig.set_size_inches(10.5, 10.5)
    ax = plt.axes(projection="3d")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    line1=ax.scatter3D(line3[:,0], line3[:,1], line3[:,2],s=18, c='r')#colormap[point_index])#colormap[categories])

    for i in range(k_centers.shape[0]):
        a=k_centers[i,0:3]
        a=np.expand_dims(a,0)
        b=k_centers[i,3:6].reshape(1,-1)
        l=creat_line(a,b,noise*0,num_points,120)
        line2=ax.scatter3D(l[:,0], l[:,1], l[:,2],s=4,c='k')#colormap[i])
        pred_line3.append(l)
       
    plt.legend((line1, line2), ('original lines', 'detected lines'))
    plt.savefig(name)
    plt.show()
    if make_gif:
        step=4
        for angle in range(0,360,step):
        
            ax.view_init(elev=10., azim=angle)
            plt.savefig("movie%d.png" % angle)
        
        outfilename = "my.gif"           
        frames = []
        for i in range(0,360,step):               
            im = imageio.imread('movie'+str(i)+'.png')            
            
            frames.append(im)                      
        imageio.mimsave(outfilename, frames, 'GIF',
                        duration=0.1)
#%%

N=540
num_lines=10
num_points=20 #number of points in each line
s=5 #scalar used to draw lines
noise=np.random.rand(num_points,3)*2
print(noise)

draw=1
# sampling azimuth and elevatuon angles
az = np.deg2rad(np.linspace(1,360,N))#azimuth
el = np.deg2rad(np.linspace(1,180,N/2))#elevation
line3,a_list,b_list=creat_toy_data(num_lines,num_points,draw,noise)
count_angle=np.array(list(itertools.product(az,el)))
cart=sph2cart(count_angle[:,0],count_angle[:,1])

num_point_threshold=20
point_distance_threshold=1.5
merge_angle_point=0
#%%
centers=hough3(line3,cart,num_point_threshold,point_distance_threshold,merge_angle_point)
colormap = np.array(['b','g','r','deeppink','y','c','m','gold','purple','pink','tan','lightgreen','sandybrown','olive'])
kmeans = KMeans(n_clusters=num_lines).fit(centers)
k_centers=kmeans.cluster_centers_
pred_line3=[]
point_index,Q1S,Q2S=cluster_points(k_centers,line3)
make_image(line3,k_centers,point_index,False,'C:/Users/Administrator/Desktop/result/7.png')
