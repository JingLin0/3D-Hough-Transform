angle_samples=540;
num_lines=4;% number of lines in the toy data
N=20;% number of points in each line
s=5; % scalar used to draw lines
line3=zeros(num_lines*N,3);
noise=rand(N,3);
for i = 1:num_lines
    temp=round(creat_line(20*rand(1,3),20*rand(1,3),noise,N,s));
    line3(((i-1)*N+1):(i*N),:)=temp;
end
scatter3(line3(:,1),line3(:,2),line3(:,3))
[acc,count_angle]=hough3(line3,angle_samples);
%%
%point
point_distance_threshold=1;
wraper=@(x)points_group(x,point_distance_threshold);
[d,C_points]=cellfun(wraper,acc,'un',0);
wrapper=@(x)size(x);
C_size=cell2mat(cellfun(wrapper,C_points,'un',0));
%%
points_group(acc{1},point_distance_threshold);
%%
%angles
angle_distance_threshold=0.01;
num_max=2;
num_pred_lines=2;

index_angle=find_top_max(C_size(:,1),num_max);
angles=count_angle(index_angle,:);
[angle_distance,angle_cluster]=points_group(angles,angle_distance_threshold);
whole_cluster=get_whole_cluster(angle_distance);
cluster_angle_point_cell={};
count=1;
for i=1:length(whole_cluster)
angle_index=index_angle(whole_cluster{i});
cccc=C_points(angle_index);
temp_cell=get_angle_point_cell(cccc,1,angle_index,C_points,count_angle);
cluster_angle_point_cell(count:(count+size(temp_cell,1)-1),:)=temp_cell;
count=count+size(temp_cell,1);
end
wrapper=@(x)size(x);
cluster_angle_point_cell_size=cell2mat(cellfun(wrapper,cluster_angle_point_cell(:,2),'un',0));
cluster_angle_point_max=find_top_max(cluster_angle_point_cell_size(:,1),num_pred_lines);
figure
scatter3(line3(:,1),line3(:,2),line3(:,3))
hold on
length(cluster_angle_point_max)

%%
filename = ['newline3D_hough1.avi'];
dtheta = 1; % Must be an integer factor of 360
v = VideoWriter(filename,'Motion JPEG AVI');
open(v);
% add first plot in 2 x 1 grid
subplot(1,2,1);
scatter3(line3(:,1),line3(:,2),line3(:,3))
ylabel('y');
xlabel ('x');
zlabel ('z');
title('original lines')
box 'on'
axis square;
set(gca,'Ticklength',[0 0])
% add second plot in 2 x 1 grid
view(3)
set(gca,'CameraViewAngleMode','Manual')
for i = 1:(360/dtheta)
    camorbit(dtheta,0,'data',[0 0 1])
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame);
end
    subplot(1,2,2);
    p=scatter3(line3(:,1),line3(:,2),line3(:,3));
    ylabel('y');
    xlabel ('x');
    zlabel ('z');
    hold on
for i=1:length(cluster_angle_point_max)
    a_p=cluster_angle_point_cell(cluster_angle_point_max(i),:) ;
    rad=mean(a_p{1},1);
    [x,y,z]=sph2cart(rad(1),rad(2),1);
    pred_line=creat_line(mean(a_p{2}),[x,y,z],noise,N,20*s);
    scatter3(pred_line(:,1),pred_line(:,2),pred_line(:,3),4)
    hold on
end
hold off

max_acc=unique(cluster_angle_point_cell_size(cluster_angle_point_max));
    title(['detected lines with acc=', num2str(max_acc(1)),' and ', num2str(max_acc(2))])
    box 'on'
    axis square;
    set(gca,'Ticklength',[0 0])
    set(gcf,'color','w');
    view(3)
    set(gca,'CameraViewAngleMode','Manual')
    for i = 1:(360/dtheta)
        camorbit(dtheta,0,'data',[0 0 1])
        drawnow
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
    close(v)

%%
function max_list= find_top_max(size_array,num_max)
[m,i]=max(size_array);
num_m=num_max;
max_list=[];
count=1;
while num_m
    temp=find(size_array==(m-count+1));
    count=count+1;
    if isempty(temp)
        continue
    else
    max_list =[max_list;temp];
    num_m=num_m-1;
    end
end
end

function angle_point_cell=get_angle_point_cell(cccc,point_distance_threshold2,angle_index,C_points,count_angle)
cccc_first=zeros(length(cccc),3);
for i=1:length(cccc)
    cccc_first(i,:)=cccc{i}(1,:);
end
[cccc_distance,cccc_cluster]=points_group(cccc_first,point_distance_threshold2);
cccc_whole_cluster=get_whole_cluster(cccc_distance);
angle_point_cell=cell(length(cccc_whole_cluster),2);
for i=1:length(cccc_whole_cluster)
angle_point_cell(i,1)={count_angle(angle_index(cccc_whole_cluster{i}),:)};
temp2=C_points(angle_index(cccc_whole_cluster{i}),:);
points=[];
for j=1:length(temp2)
    points=[points;temp2{j}];
end
angle_point_cell(i,2)={points};
end
end

function whole_cluster=get_whole_cluster(distances_binary)
sum_distances=sum(distances_binary,1);
whole_cluster={};
while norm(sum_distances)>0
[m_s,index_s]=max(sum_distances);
index_same=find(distances_binary(:,index_s)==1);
whole_cluster{end+1}=index_same;
sum_distances(index_same)=0;
end
end


function [distances_binary,point_cluster]=points_group(acc_cell,distance_threshold)
num_xx=size(acc_cell,1);
distances=ones(num_xx,num_xx);
distances_binary=zeros(num_xx,num_xx);
for i =1:num_xx
    temp=repmat(acc_cell(i,:),num_xx,1);
    distances(i,:)=sum((temp-acc_cell).^2,2).^(1/2);
end
distances_binary(distances<=distance_threshold)=1;
sum_distances=sum(distances_binary,1);
[m_s,index_s]=max(sum_distances);
%index_s=sum_distances==m_s;
index_same=distances_binary(:,index_s)==1;
point_cluster=acc_cell(index_same,:);
size(point_cluster)

end

function [acc,count_angle] = hough3(p,angle_samples)
% Input
% p? the coordinates of all collinear points
%angle samples: the number of angles we sample
% Output:
% acc: store all the coordinates of perpendicular foot. : (angle_samples^2)/2*numOfPoints*3
%count_angle: all the combinations of angles : (angle_samples^2)*2

N = angle_samples;
az = deg2rad(linspace(1,360,N));% azimuth
el = deg2rad(linspace(1,180,N/2));% elevation
count_angle=zeros(length(az)*length(el),2);
acc=cell(N*(N/2),1);
count=0;
for i = 1:length(az)
    for j = 1:length(el)
        count=count+1;
        count_angle(count,:)=[az(i),el(j)];
        x=cos(el(j))*cos(az(i));
        y=cos(el(j))*sin(az(i));
        z=sin(el(j));
        v=[x,y,z];% direction vector
        A=p;%numOfPoints*3
        B=p+v; %a line with points A and B
        origin=[0,0,0];
        k=((origin-A)*v')./(sum(abs(B-A).^2,2));%numOfPoints*3
        p_prime=k.*(B-A)+A;%find the perpendicular foot coordinate
        key_temp=round(p_prime(:,1:3),1);% numOfPoints*3
        acc{count,1}=key_temp;
    end
end
whos ___ acc
end
function line3D=creat_line(a,b,noise,num_points,scale)
%creat 3d line line=a+tb+noise
%Input:
%a: a point on the line shape 1*3
%b: direction vector shape 1*3
%noise: random noise shape numberOfPoints*3
%scale: scalar t
%Output:
%3D line coordinates
b_matrix=repmat(b,num_points,1);
scalar=scale*rand(num_points,1);
scalar_matrix=[scalar,scalar,scalar];
line3D=(a+b_matrix.*scalar_matrix+noise);
end
