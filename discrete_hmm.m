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
O = 2; % num observable symbols

ns = [Q O];
dnodes = 1:2;
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes);
for i=1:4
  bnet.CPD{i} = tabular_CPD(bnet, i);
end
