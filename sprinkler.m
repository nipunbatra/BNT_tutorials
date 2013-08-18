%% Introduces the rain-sprinkler example

% Creating the adjacency matrix
N = 4; 
dag = zeros(N,N);
C = 1; S = 2; R = 3; W = 4;
dag(C,[R S]) = 1;
dag(R,W) = 1;
dag(S,W)=1;

% Saving the DAG
h=figure;
draw_graph(dag);
saveas(h,'sprinkler','jpg');
close(h);

%Creating the BNT shell
discrete_nodes = 1:N;
node_sizes = 2*ones(1,N); 

bnet = mk_bnet(dag, node_sizes, 'names', {'cloudy','S','R','W'}, 'discrete', 1:4);

% Specifying the CPTs
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

% Instantiating the Junction-Tree inference engine
engine = jtree_inf_engine(bnet);

%%% Now we will inject evidence and compute marginal distibutions %%%

%%% ----------------------------------------------------------------------------------
%%% CASE 0: Find the probability that the sprinker was on given no evidence
%%% This should be P(C=F,S=T) + P(C=T, S=T) =.5*0.5 + 0.5*0.1= 0.3
%%%-----------------------------------------------------------------------------------
engine = jtree_inf_engine(bnet);

evidence = cell(1,N);
evidence{W} = [];

[engine, loglik] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, S);
no_evidence = marg.T(2);

%%% ----------------------------------------------------------------------------------
%%% CASE I: Find the probability that the sprinker was on given that the grass is wet
%%%-----------------------------------------------------------------------------------
engine = jtree_inf_engine(bnet);
evidence=cell(1,N);
evidence{W}=2;
[engine, loglik] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, S);
evidence_grass_wet = marg.T(2);

%%% ----------------------------------------------------------------------------------
%%% CASE II: Find the probability that the sprinker was on given that the grass is not wet
%%%-----------------------------------------------------------------------------------
engine = jtree_inf_engine(bnet);
evidence=cell(1,N);
evidence{W}=1;
[engine, loglik] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, S);
evidence_grass_not_wet = marg.T(2);

%%% ----------------------------------------------------------------------------------
%%% CASE III: Find the probability that the sprinker was on given that the
%%% grass is wet and it did not rain
%%% We would expect this probability to be very high, since now sprinkler
%%% only should cause grass to be wet
%%%-----------------------------------------------------------------------------------
engine = jtree_inf_engine(bnet);
evidence=cell(1,N);
evidence{W}=2;
evidence{R}=1;
[engine, loglik] = enter_evidence(engine, evidence);
marg = marginal_nodes(engine, S);
evidence_grass_wet_no_rain = marg.T(2);

h=figure;
cases={'No evidence','G=dry','G=wet','G=wet, R=false'};
set(0,'DefaultAxesFontSize',30)
bar([1:4],[no_evidence,evidence_grass_not_wet,evidence_grass_wet,evidence_grass_wet_no_rain]);
set(gca,'XtickL',cases);
title('P(Sprinkler=On|Evidence)');
xlabel('Evidence');
ylabel('P(Sprinkler=On)');

