%% Setting up the path
cd ~/git/bnt_tutorial/
addpath(genpath(pwd))

clear;

% Maintaining a constant seed for experiment reproducibility
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

%Now using the parameters which we had learnt before
learnt_transmat=[0.74,0.26;0.3,0.7];
learnt_prior=[0.11;0.89];
learnt_obsmat=[0.178052578451625,0.173883549445059,0.163420227386818,0.190515178728016,0.169639153430181,0.124489312558300;0.107532640659765,0.104160605453397,0.0977215094092837,0.106901812648972,0.0991482097246066,0.484535222103976];

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
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes,'observed',onodes, 'eclass1', eclass1, 'eclass2', eclass2);

%% We define the HMM parameters as follows
% Ordering is as follows: Fair, Unfair

bnet.CPD{1} = tabular_CPD(bnet, 1, learnt_prior);
bnet.CPD{2} = tabular_CPD(bnet, 2, learnt_obsmat);
bnet.CPD{3} = tabular_CPD(bnet, 3, learnt_transmat);

T = 1000;
ncases=;
cases = cell(1, ncases);
ss=2;
max_iter=10;
for i=6:6+ncases
  cases{i} = cell(ss,T);
  filename=sprintf('data/observed_%d.csv',i);
  filename_hidden=sprintf('data/hidden_%d.csv',i);
  data=csvread(filename);
  ground_truth=csvread(filename_hidden);
end
obslik = multinomial_prob(data, learnt_obsmat);
path = viterbi_path(learnt_prior, learnt_transmat, obslik);

%% Plotting the results
subplot(211)
plot(path)
ylim([0.8 2.2])
title('Ground Truth');
subplot(212)
plot(ground_truth)
ylim([0.8 2.2])
title('Predicted using Viterbi algorithm');

%%Computing statistics regarding the prediction
accuracy=sum(path==ground_truth)/length(path);