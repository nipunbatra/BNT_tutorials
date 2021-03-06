% Make an HMM with Gaussian observations
%   X1 -> X2
%   |     | 
%   v     v
%   Y1    Y2 
%% Setting up the path
cd ~/git/bnt_tutorial/
addpath(genpath(pwd))

%% Setting a seed for repeatibility
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
clear;
intra = zeros(2);
intra(1,2) = 1;
inter = zeros(2);
inter(1,1) = 1;
n = 3;

Q = 2; % num hidden states
O = 1; % size of observed vector
ns = [Q O];
bnet = mk_dbn(intra, inter, ns, 'discrete', 1, 'observed', 2);

prior0 = [0.8 0.2];
transmat0 = [0.9 0.1; 0.1 0.9];
mu0 = [0 160];
Sigma0 = repmat(eye(O), [1 1 Q]);
bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
%% we set the cov prior to 0 to give same results as HMM toolbox
%bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', mu0, 'cov', Sigma0, 'cov_prior_weight', 0);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', mu0, 'cov', Sigma0);
bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);

T=1000;
ev = sample_dbn(bnet, T);

%% Stats regarding the produced distribution

hid=cell2mat(ev(1,:));
obs=cell2mat(ev(2,:));


engine = {};
engine{end+1} = smoother_engine(jtree_2TBN_inf_engine(bnet));
engine{end+1} = smoother_engine(hmm_2TBN_inf_engine(bnet));
engine{end+1} = hmm_inf_engine(bnet);
engine{end+1} = jtree_unrolled_dbn_inf_engine(bnet, T);
%engine{end+1} = frontier_inf_engine(bnet);
engine{end+1} = bk_inf_engine(bnet, 'clusters', {[1]});
engine{end+1} = jtree_dbn_inf_engine(bnet);


inf_time = cmp_inference_dbn(bnet, engine, T);

ncases = 2;
max_iter = 2;
[learning_time, CPD, LL, cases] = cmp_learning_dbn(bnet, engine, T, 'ncases', ncases, 'max_iter', max_iter);

% Compare to HMM toolbox

data = zeros(O, T, ncases);
for i=1:ncases
  data(:,:,i) = cell2num(cases{i}(bnet.observed, :));  
end

tic
[LL2, prior2, transmat2, mu2, Sigma2] = mhmm_em(data, prior0, transmat0, mu0, Sigma0, [],  'max_iter', max_iter);
t=toc;
disp(['HMM toolbox took ' num2str(t) ' seconds '])

e = 1;
assert(approxeq(prior2, CPD{e,1}.CPT))
assert(approxeq(mu2, CPD{e,2}.mean))
assert(approxeq(Sigma2, CPD{e,2}.cov))
assert(approxeq(transmat2, CPD{e,3}.CPT))
assert(approxeq(LL2, LL{e}))
subplot(211)
 plot(cell2mat(ev(1,:)));figure(gcf);
subplot(212)
plot(cell2mat(ev(2,:)));figure(gcf);
