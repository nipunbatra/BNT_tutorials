% Make an HMM with discrete observations
%   X1 -> X2
%   |     | 
%   v     v
%   Y1    Y2 

intra = zeros(2);
intra(1,2) = 1;
inter = zeros(2);
inter(1,1) = 1;
n = 2;

Q = 2; % num hidden states
O = 6; % num observable symbols

ns = [Q O];
dnodes = 1:2;
onodes = [2];
eclass1 = [1 2];
eclass2 = [3 2];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, ...
	      'observed', onodes);

rand('state', 0);
prior1 = normalise(rand(Q,1));
transmat1 = mk_stochastic(rand(Q,Q));
obsmat1 = mk_stochastic(rand(Q,O));
bnet.CPD{1} = tabular_CPD(bnet, 1, prior1);
bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat1);
bnet.CPD{3} = tabular_CPD(bnet, 3, transmat1);
T = 30
engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
ncases=10
cases = cell(1, ncases);
ss=2;
for i=1:ncases
  ev = sample_dbn(bnet, 'length', T);
  cases{i} = cell(ss,T);
  cases{i}(onodes,:) = ev(onodes, :);
end
  [bnet2, LL] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);
   prior=struct(bnet2.CPD{1})
   prior.CPT
   prior1
   
   tran=struct(bnet2.CPD{2})
   tran.CPT
   transmat0
   
   
