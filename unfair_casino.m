%% Setting up the path
cd ~/git/bnt_tutorial/
addpath(genpath(pwd))

%% Setting a seed for repeatibility
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

%% Creating the DBN node structure
intra = zeros(2);
intra(1,2) = 1; % node 1 in slice t connects to node 2 in slice t

inter=zeros(2);
inter(1,1)=1; % node 1 in slice t-1 connects to node 1 in slice t

%% Specifying the observed and hidden states
Q = 2; % num hidden states
O = 6; % num observable symbols

%% Defining equivalence classes
%% Class 1; first node- P(class1) depends on prior only
%% Class 2; observed nodes- P(class 2) depends on emission only
%% Class 3; hidden nodes- P(class 3) depends on transition only

eclass1 = [1 2];
eclass2 = [3 2];
eclass = [eclass1 eclass2];

ns = [Q O];
dnodes = 1:2;
onodes = [2];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

%% We define the HMM parameters as follows
% Ordering is as follows: Fair, Unfair
% Pi= [2/3,1/3]
% A= [[0.95,0.05],[0.1,0.9]]
% B= [[1/6,1/6....1/6],[1/5,1/5,1/5,1/5,1/5,1/2]]
prior0 = [.67;.33];
transmat0 = [[0.95,0.05];[0.1,0.9]];
obsmat0 = [1/6*ones(1,6);[1/10,1/10,1/10,1/10,1/10,1/2]];
bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat0);
bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);

%% Sampling from the DBN
T=1000;
ev = sample_dbn(bnet, T);

%% Stats regarding the produced distribution

hid=cell2mat(ev(1,:));
obs=cell2mat(ev(2,:));

% Counting the number of occurences of each face
count_obs=zeros(1,6);
count_fair=zeros(1,6);
count_biased=zeros(1,6);
for i=1:6
   count_obs(i)=sum(obs==i);
end

% Counting the number of occurences of fair and biased throw
count_hid=[sum(hid==1) sum(hid==2)];
 
% Frequency by fair/biased
for i=1:6
    count_fair(i)=sum((obs==i) & (hid==1));
    count_biased(i)=sum((obs==i)&(hid==2));
end

%Setting font size
set(0,'DefaultAxesFontSize',20)

h=figure;
sub1=subplot(221);
bar(count_fair);
grid();
title(sub1,'Frequency of die faces (Fair)');
sub2=subplot(223);
bar(count_biased);
grid();
title(sub2,'Frequency of die faces (Biased)');
sub3=subplot(222);
bar(count_obs);
grid();
title(sub3,'Frequency of die faces (overall)');
sub4=subplot(224);
bar(count_hid);
grid();
title(sub4,'Frequency of biased and fair die');
set(gca,'XtickL',{'fair','biased'});
close(h);

% Plotting the distribution  

h=figure;
s1=subplot(211);
area(hid);
title(s1,'Die Nature (Fair=1; Biased=2)');
ylim([1,2]);
s2=subplot(212);
plot(obs,'go');
title(s2,'Die faces');
ylim([0.5,6.5]);
grid();

% Saving the data in 10 files (one each for hidden and one for 
% observed, numbered from 1 through 10
for i=1:10
    ev = sample_dbn(bnet, T);
    hid=cell2mat(ev(1,:));
    obs=cell2mat(ev(2,:));
    name=sprintf('data/hidden_%d.csv',i);
    csvwrite(name,hid);
    name=sprintf('data/observed_%d.csv',i);
    csvwrite(name,obs);
end
    
    
    


