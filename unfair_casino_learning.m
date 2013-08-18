clear;

%% Maintaining a constant seed for experiment reproducibility
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
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes,'observed',onodes, 'eclass1', eclass1, 'eclass2', eclass2);

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

T = 500
engine = smoother_engine(hmm_2TBN_inf_engine(bnet));
% engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
ncases=2
cases = cell(1, ncases);
ss=2;
max_iter=10;
for i=1:ncases
  ev = sample_dbn(bnet, 'length', T);
  cases{i} = cell(ss,T);
  cases{i}(onodes,:) = ev(onodes, :);
end
  [bnet2, LL] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);