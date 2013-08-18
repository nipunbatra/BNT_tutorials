%% Setting up the path
cd ~/git/bnt_tutorial/
addpath(genpath(pwd))

%% Creating the DBN node structure
intra = zeros(2);
intra(1,2) = 1; % node 1 in slice t connects to node 2 in slice t

inter=zeros(2);
inter(1,1)=1; % node 1 in slice t-1 connects to node 1 in slice t

%% Specifying the observed and hidden states
Q = 2; % num hidden states
O = 6; % num observable symbols

%% Defining equivalence classes
eclass1 = [1 2];
eclass2 = [3 2];
eclass = [eclass1 eclass2];

ns = [Q O];
dnodes = 1:6;
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);
prior0 = normalise(rand(Q,1));
transmat0 = mk_stochastic(rand(Q,Q));
obsmat0 = mk_stochastic(rand(Q,O));
bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat0);
bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);

%% Sampling from the DBN
T=100;
ev = sample_dbn(bnet, T);

