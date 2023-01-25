using Distributions, Plots, Statistics, StatsBase, Random, DataFrames, Tables, SQLite, ProgressLogging, StatsPlots

############################ Key Assumptions ###################
# Individuals start with shared uniform distribution over proportions of accept/reject
# Individuals receive common signal from vector of performances
# Individuals have symmetrical preferences over accept and reject. Strong accept|accept preference = strong and opposite in sign R|R preference
# No individual heterogeneity in terms of the cost of being in the minority
# Three preference classes exist: Prefer R|R (uniform from -1 to 0), Prefer A|A (uniform from 0 to 1) and unsure (U(A|A) = U(R|R) = 0)
#       Why not distributed normally within each class?    
# Sequence of statements is fixed and exogenous. Individuals MUST state, cannot hide their statement. 
# Use mode of posterior not mean
# If agents have identical utilites to A|A and R|R, pick whichever their true preference is (idea being that they understand possibility they might influence others)
############################ TO DO ##########################
# Should agent which prefers Accept have any/negative preference for reject? Makes sense to me 
# Should agents choose when to speak? Can play a dynamic game. 

# how do novel behaviors become classified? What are the dynamics of even a sequential process whereby individuals are required to approve or disapprove of a novel behavior, and do not want to end up in the minority (some have a private interest in approval, some in disapproval.)

# I don’t see social learning in this context—other than learning about what the sequence of announcements to date is. I’m thinking of just individual decisionmaking—simplest version: across different random orderings (but perhaps always led by one who prefers ‘acceptable’ label for a new action; and then maybe a bias towards those with an interest voicing early views because they have an incentive to influence positions of later agents) what is the optimal decision for each agent in the sequence (approve/disapprove) given an assumed cost to being in the minority once consensus is reached (ie. point at which agents all announce the same thing).  I’m assuming this is a simulation, exploring the impact of varying parameters.  In a sense, this is what Atrisha is looking at—but her version is a bit more complex. We’d be looking at how long it takes consensus to emerge, whether novel behaviors are approved or disapproved based on differing sizes of groups, sequencing, etc.  

######
# A group of size N exists. This group is organized around a classifier which assigns each behavioral performance a 1 or -1 denoting normatively acceptable or not. A new performance has occurred which must be classified. Individuals have opinions on whether new behavior should be classified as norm adhering (1) or norm breaking (-1). Individuals possess preferences over both the outcome and being in a majority. Preferences take the form of the utility of stating accept given that the group ultimately accepts, stating reject given that the group ultimately rejects, and miscoordinating with the group. 
# The sequence of decision occurs either completely randomly or stochastically whereby agents with more extreme opinions are more likely to state them earlier in the sequence. Agents cannot avoid stating their opinion when it is their turn to do so. As each individual observes the sequence of public statements or performances they update their beliefs that the group will end up picking to either accept or reject the proposal. These beliefs are either binomial (accept reject) or multinomial (acc, rej, unsure, i.e. the number of agents remaining to make their declaration). 
# Finally, once all agents have made their statements the group selects an outcome. If agents stated accept and the group accepts, they receive utilAccept. If agents stated reject and the group rejects, they receive -utilAccept (i.e., u(R|R)). If agents are in the minority and miscoordinate with the group, they receive -minorityCost. This is a commonly experienced cost regardless of agent's statement (i.e. no difference if agent Accepted and group ultimately Rejects or vice versa). 

######
# Future Ideas
# 1. Agents experience heterogenous costs to being in minority (some individuals don't care)
# 2. Endogenize cost to being in the minority. Would need to think what is actually at cost and how that would vary among individuals.
# 3. Individuals possess distinct prior probabilities. While they receive common signal, their initial decisions will be distinct.
# 4. Should individuals be able to change their opinions after they've made a statement?

mutable struct Agent
    id::Int64                   # Identifier
    utilAccept::Float16         # Utility of A|A ∈ (-1, 1) where 0 denotes ambiguouity or unsure, -1 denotes normatively unacceptable; 1 denotes normatively acceptable. 
                                # Values closer to zero denote weaker directional preferences. A 0 means you get no utility from accept|A or R|R and you only care about avoiding minority. 
    decision::Int               # What does agent ultimately state: 1 = accept, 2 = reject
    utility::Float16            # Final utility of agent
    beliefAccept::Float32       # Belief agent encounters that the group will accept
end

mutable struct Society
    N::Int64                                        # Number of individuals in group
    agents::Vector{Agent}                           # Vector of all Agents
    utilities::Vector{Float16}                      # Vector of utilities that agents incur
    performances::Vector{Int64}                     # Vector of stated performances (1 = Accept, -1 = Reject)
    randomSeq::Bool                                 # True if random sequence of statements otherwise false
    propPref::Vector{Float16}                       # Proportion of accept, unsure, reject
    minorityCost::Float16                           # Cost to being in minority ultimately
    binomialBeliefs::Bool                           # True if agents have binomial beliefs, 
                                                    # false if agents have multinomial
    sharedBeliefs::Vector{Int64}                    # Distribution of beliefs defined by parameters a [1], b [2]
end

""" Function initializes Society
    User must supply with N, group size; minorityCost, cost of being in minority at end; randomSeq, whether performances are stated in random order (true) or proportionate to true preference over outcomes (false); propPref, vector of three classes definining proportion of group with preferences that prefer proposal is [rejected ∈ (-1, 0), unsure about = 0, accepted ∈ (0, 1)]; binomialBeliefs, whether beliefs are binomial and Beta (true) or multinomial & Dirichlet (false).
    1. Assign utility: propPref, three classes, disapprove, unsure, approve. Sample proportional to propPref (vector of rejecting, middle, acceptance)
    2. Populates with N agents
    3. Initializes sequence of performances: if randomSeq, random else proportional to utilities (more extreme opinions go first)
    4. Assigns initial beliefs: Beta if true and Dirichelt (multinomial) if false. 
"""
function init(N, minorityCost, randomSeq, propPref, binomialBeliefs)
    if sum(propPref) != 1
        throw("Preference classes do not sum to 1.")
    end

    # 1. Assign utility for A|A to each agent (Utility of R|R = -U(A|A), a strong pref for A means strong dislike of R)
    prefs = [Distributions.Uniform(-1, 0), Distributions.Normal(0, 0), Distributions.Uniform(0, 1)] # Three classes of agents - reject (-1,0), unsure (0), approval (0, 1)
    
    # 2. Populate with N agents
    agents = [Agent(i, rand(StatsBase.wsample(prefs, Weights(propPref)) ), 0, 0.0, .5) for i in 1:N] # ID, preference, NO DECISION YET, NOR UTILITY, belief (init to .5)
    
    # 3. Initialize sequence of performances
    utilsTransformed = abs.(getfield.(agents, :utilAccept)) .+ .001 # Calculate distance from 0 with small amount added to permit sampling of unsure individuals
    if randomSeq # Sequence is random
        agents = Random.shuffle(agents)
    else # Sequence is non-random but order is ∝ magnitude of preference over outcomes.
        agents = StatsBase.wsample(agents, Weights(utilsTransformed./sum(utilsTransformed)), N, replace = false)
    end
    
    # 4. Initialize beliefs as a uniform distribution of some sort
    # If beliefs are binomial, prior is beta on proportion of accept, else beliefs are multinomial and prior is Dirichlet
    sharedBeliefs = ifelse(binomialBeliefs, [1,1], [1,1,1]) # No prior knowledge, no sampling

    # Assign empty vectors
    utilities = []
    performances = []
    return Society(N, agents, utilities, performances, randomSeq, propPref, minorityCost, binomialBeliefs, sharedBeliefs)
end

"""
    Function proceeds through previously defined sequence stored in Society struct
    1. Each agent faces decision contingency defined by coordination issue where they want to avoid the minority
        p_A is belief that group will select Accept as majority
        Cost to being in minority: -c, minorityCost
        Agent's degree of preference that proposal is accepted given that group ultimately accepts: utilAccept ∈ (-1, 1)
        Agent's degree of preference that proposal is rejected given that group ultimately rejects: -utilAccept (This means a strong preference for accept|accept means a strong rejection for reject|reject. A weak preference for accept means a weak preference for reject. This seems a reasonable way to define utility)
            Utility of accept = p_A*u_A^i + (1-p_A)(-c): If I state accept then with probability group will accept I get utilAccept (u_A^i). With 1-p_A I miscoordinate and am in the minority, getting -c
            Utility of reject = p_A*(-c) + (1-p_A)(-u_A^i): If I state reject then with probability (1-p_A) the group will also reject I get -utilAccept (-u_A^i). With p_A I miscoordinate and am in the minority (I stated reject but group stated accept), getting -c
    2. Append statement to public sequence & update beliefs
    3. Append to dataframe
    4. Assign utility
"""
function iterate!(soc::Society)
    df = DataFrame(Agent = Int64[], AlphaBefore = Int64[], BetaBefore = Int64[], uAA = Float16[], statement = Int64[] , AlphaAfter = Int64[], BetaAfter = Int64[], MinorityCost = Float16[], BeliefAccept = Float32[], Threshold = Float32[], UtilMinusThreshold = Float32[], SeqPosition = Int64[])
    # Iterate through agents
    sequence_counter = 1
    for agent in soc.agents

        # 1. Decision contingency

        # Get parameters defining belief
        a = soc.sharedBeliefs[1] # Number accepts
        b = soc.sharedBeliefs[2] # Number rejects
        # First check if previous performances and nature of beliefs - binomial or not
        if soc.binomialBeliefs
            mode_accept = ifelse(a == 1 && b == 1, 1/2, (a-1)/(a+b-2)) # If both params are 1, we have a uniform.
            mode_reject = 1-mode_accept
        else
            c = soc.N - a - b                                                      # Number of remaining, undecided individuals
            mode_accept = ifelse(a == 1 && b > 1 && c == 1, 1/3, (a-1)/(a+b+c-3))    # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do
            mode_reject = ifelse(a == 1 && b == 1 && c == 1, 1/3, (b-1)/(a+b+c-3))    # If a, b, c are all 1 then it's a uniform distribution and 1/3 is the best you'll do
        end
        agent.beliefAccept = mode_accept
        # Each agent compares the utility of accepting or rejecting given beliefs which is a coordination game with the majority coalition
        # Utility of stating accept is the preference for accept * prob accept - cost of minority * probability reject (because focal agent stated accept)
        U_A = agent.utilAccept*mode_accept - soc.minorityCost*mode_reject
        # Utility of stating reject is the preference for reject (the negative of utilAccept) * prob reject - cost of minority * probability accept (because focal agent stated reject)
        U_R = -agent.utilAccept*mode_reject - soc.minorityCost*mode_accept
            
        # Set agent's decision: If one of actions has a higher utility, select that, else if (utilities of actions are same, but agent has true pref), select true preference, else (agent has no true pref), select randomly
        if U_A != U_R
            agent.decision = ifelse(argmax([U_A, U_R]) == 1, 1, -1) # First is accept then reject, set decision
        elseif agent.utilAccept != 0
            agent.decision = ifelse(argmax([agent.utilAccept, -agent.utilAccept]) == 1, 1, -1) # If utilities of each action are same, pick A if UAA higher than URR. Idea here is that agents have some idea they're trying to persuade. 
        else # if agent has U_A = U_R AND utilAccept = 0 (they have no preference) then pick randomly
            agent.decision = ifelse(rand([1, -1]) == 1, 1, -1)
        end
        # 2. Append statement to public sequence and update beliefs
        push!(soc.performances, agent.decision) # Append to array

        # 3. Update sharedBeliefs

        if agent.decision == 1                      # If agent accepted then add to α spot
            soc.sharedBeliefs[1] += 1
            
        elseif agent.decision == -1                 # If agent rejected, then add 1 to β spot
            soc.sharedBeliefs[2] += 1
        
        elseif soc.binomialBeliefs == false         # If multinomial beliefs, knock 1 from final unsure spot
            soc.sharedBeliefs[3] -= 1
        end
        
        Threshold = (1/2)*(1-agent.utilAccept/soc.minorityCost)
        UtilMinusThreshold = agent.utilAccept - soc.minorityCost*(1-2*agent.beliefAccept)
        # 3. Fill in Df with agent ID, AlphaBefore, BetaBefore, utilAccept, decision, AlphaAfter, BetaAfter, minorityCost, and belief (for ease)
        push!(df, Dict(:Agent => agent.id, :AlphaBefore => a, :BetaBefore => b, :uAA => agent.utilAccept, :statement => agent.decision, :AlphaAfter => soc.sharedBeliefs[1], :BetaAfter => soc.sharedBeliefs[2], :MinorityCost => soc.minorityCost, :BeliefAccept => agent.beliefAccept, :Threshold => Threshold, :UtilMinusThreshold => UtilMinusThreshold, :SeqPosition => sequence_counter))

        sequence_counter += 1
    end
    
    # 4. Assign utility
    # All agents have made statements, find majority coalition
    finalAccept = sum(soc.performances .== 1)
    finalReject = sum(soc.performances .== -1)
    proportionAccept = finalAccept/(finalAccept + finalReject) # Size of accept coalition
    proportionReject = finalReject/(finalAccept + finalReject) # Size of reject coalition

    # Any individual who stated opposite of majority gets -c 
    # Any individual who stated in line with majority gets u_A^i or -u_A^i

    coalition = ifelse(argmax([proportionAccept, proportionReject])==1, 1, -1)
    for agent in soc.agents
        if coalition == 1 && agent.decision == coalition        # If group selects accept and agent prefers accept
            agent.utility += agent.utilAccept
        elseif coalition == -1 && agent.decision == coalition   # If group selects reject and agent prefers reject
            agent.utility += -agent.utilAccept
        else                                                    # If agent miscoordinates with group
            agent.utility += -soc.minorityCost
        end
    end

    # Transfer agents' utilities to easily accessible vector
    soc.utilities = getfield.(soc.agents, :utility) 

    # set majority so I know
    df.FinalProportion .= proportionAccept
    # 
    if proportionAccept > .5
        df.FinalOutcome .= 1
    elseif proportionAccept < .5
        df.FinalOutcome .= -1
    else
        df.FinalOutcome .= 0
    end
    # Return data frame
    return df
end

"""
    Function initializes a society and runs gen times (independent universes), storing the simulation in an SQLdatabase.
    User provides
        gens                INT                 Number of universes that should be run
        N                   INT                 Size of communities
        minorityCost        FLT                 Cost of being in the minority
        randomSeq           BOOL                true if random sequence, false if sequence prop to magnitude of pref
        propPref            ARR                 Proportion of society with preferences rejecting, unsure, and accepting
        binomialBeliefs     BOOL                true if binomial beliefs else multinomial
        tableName              TXT              Name of table inside of database, if false

    DB Stored values are: gen (INTEGER & primary key), agent (INTEGER and effectively time variable defined by sequence of choice), alpha & beta values (INTEGERS), prior (REAL) denoting alpha/(alpha + beta) prior, uAA (REAL the utility of A|A), uRR (REAL the utility of R|R), statement (INTEGER, 1 or -1 denoting what agent stated)
    
    Final database gives generation columns of sequences of 1) priors, 2) statements, 3) utilities (of agents)
"""
function run(gens, N, minorityCost, randomSeq, propPref, binomialBeliefs, tableName)
    
    if typeof(tableName) == String
        db = SQLite.DB("Sim Data/SimulationDB.db")
    end
    # CREATE TABLE I don't think I need this using DataFrames
    #SQLite.createtable!(db, "$tableName", 
    #    Tables.Schema((:Gen, :Agent, :AlphaBefore, :BetaBefore, :uAA, :statement, :AlphaAfter, :BetaAfter), 
    #        (Int64, Int64, Int64, Int64, Float16, Int64, Int64, Int64)))
    
    df = DataFrame()
    for genIter in 1:gens
        society = init(N, minorityCost, randomSeq, propPref, binomialBeliefs)
        gen_df = iterate!(society) # returns data base
        gen_df.Gen .= genIter # Stack generation number

        append!(df, gen_df) # Append to df
    end
    
    if typeof(tableName) == String
        # Convert to sql table
        SQLite.load!(df, db, "$tableName")
    end
    return df
end
df = run(1000, 40, .7, false, [.3, .5, .2], true, nothing)

# society = init(20, .8, false, [.3, .4, .3], false)
# test = iterate!(society)

# test = run(3, 20, 1.7, false, [.3, .4, .3], true, "test2")

# v = getfield.(society.agents, :utilAccept)
# mean(v.<0/1000000)
# mean(v.>0/1000000)
# mean(v.==0/1000000)
# g=[]
# for i in 1:2000
#     society = init(20, .2, false , [.3, .5, .2], true)
#     println(mean(getfield.(society.agents, :utilAccept)))
#     iterate!(society)
#     push!(g, mean(society.performances))
# end
# print(mean(g))
# histogram(g; bins = -1.1:.05:1.1)
# print(mean(society.utilities))

############################################ Helper Functions ############################################
##### Plotting 

"""
    Function plots beliefs across universes. Plots a proportion of them with a given transparency (alpha value) then plots the average conditioned on which majority wins as well as the marginalized average. 
    User supplies DF of simulated data output by run function previously defined, where each row is a society member's decision and alpha values, proportion of conditional distributions to plot, and whether or not the proportion refers to marginal or conditional distributions as well as other kwargs
    Function groups into outcomes (if alpha > beta, majority is ACCEPT else majority is REJECT)
    In conditional plots, function also shows the average belief an individual at that position in the sequence must have to make an accept or reject decision. This is found by calculating c(1-2p) for each agent, the threshold that must be exceeded to state ACCEPT publicly. It then plots a distribution of U_AA^i-c(1-2p^i) where U_AA^i is the ith agent's utility to ACCEPT conditional on the group accepting and p^i is there belief the these for a sample of agents at that position in the sequence.
"""
function plotting_function(df; useralpha = .3, proportion = .01, marginalproportion = true)
    # Take in data frame where each row is a universe/society, agent tuple. 
    # Get community size, N
    N = DataFrames.nrow(filter(x -> x.Gen == 1, df))
    # Get number of universes run
    Gens = maximum(df.Gen) 
    
    # Iterate through each universe, pull out sequence of beliefs, outcomes, final outcome, utility, and cost and store as named tuple in array
    seqOutcome = []
    

    @progress for univ_iter in 1:Gens # Iterate through each generation
        # Filter dataframe to given universe
        tempDF = df[df.Gen .== univ_iter, :]
        
        # Pull out sequence of statements
        # Identify majority
        majorityAccept = sum(tempDF.statement .== 1)/N

        # Store named tuple of majorityAccept, AlphaBefore, BetaBefore, statements into array
        push!(seqOutcome, (majority = majorityAccept, alphas = tempDF.AlphaBefore, betas = tempDF.BetaBefore, statement = tempDF.statement, cost = tempDF.MinorityCost, utility = tempDF.uAA, modalBelief = tempDF.BeliefAccept, Threshold = tempDF.Threshold, UtilMinusThreshold = tempDF.UtilMinusThreshold))
    end
    # Now have array of each society's outcome, trajectory of beliefs, decisions, utility of the agent making that decision, and the cost associated with the minority
       
    # Separate out accepts, rejects, and unsures for ease
    accepts = filter(x -> x.majority > .5, seqOutcome)
    rejects = filter(x -> x.majority < .5, seqOutcome)

    # Construct named tuple that has marginal and conditioned average beliefs and Threshold
    if length(accepts) != 0
        Accepts_beliefAverage = mean(map(x -> x.modalBelief, accepts)) 
        Accepts_thresholdAverage = mean(map(x -> x.Threshold, accepts))
    else
        Accepts_beliefAverage = NaN
        Accepts_thresholdAverage = NaN
    end
    if length(rejects) != 0
        Rejects_beliefAverage = mean(map(x -> x.modalBelief, rejects))
        Rejects_thresholdAverage = mean(map(x -> x.Threshold, rejects))

    else
        Rejects_beliefAverage = NaN
        Rejects_thresholdAverage = NaN
    end

    beliefsAverage = (
                        beliefAverage = mean(map(x -> x.modalBelief, seqOutcome)), 
                        thresholdAverage = mean(map(x -> x.Threshold, seqOutcome)),

                        # Conditioned on majority Accept
                        Accepts_beliefAverage = Accepts_beliefAverage, 
                        Accepts_thresholdAverage = Accepts_thresholdAverage,
                        

                        # Conditioned on majority Reject
                        Rejects_beliefAverage = Rejects_beliefAverage, 
                        Rejects_thresholdAverage = Rejects_thresholdAverage
    ) 


    # Pull out threshold sequence for each universe (array of sequences)
    # N x Gens matrix, each column is a series of thresholds
    thresholdsOverall = hcat(map(x -> x.Threshold, seqOutcome)...) 
    # Pull out belief sequence for each universe (array of sequences)
    # N x Gens matrix, each column is a series of modal beliefs (N long)
    beliefsOverall = hcat(map(x -> x.modalBelief, seqOutcome)...)
    beliefsThreshDif_Overall = thresholdsOverall - beliefsOverall

    # Do same for accept and reject subgroups
    # Accepts
    thresholdsAccept =  hcat(map(x -> x.Threshold, accepts)...)
    beliefsAccept = hcat(map(x -> x.modalBelief, accepts)...)
    beliefsThreshDif_Accepts = thresholdsAccept - beliefsAccept
    # Rejects
    thresholdsReject =  hcat(map(x -> x.Threshold, rejects)...)
    beliefsReject = hcat(map(x -> x.modalBelief, rejects)...)
    beliefsThreshDif_Rejects = thresholdsReject - beliefsReject

    # Now have distribution of threshold values, beliefs, and the difference between them for each agent position across all universes in ≤N x Gens matrices
    ########################## Plots ##########################
    ########### Avg and Conditional Beliefs Trajectory
    ###### Avg plot (1)
    # Plot will be a (2,1) with the top plot showing the average marginal beliefs and the bottom plot conditioning on accept and reject

    # Sample user-provided proportion outcomes to plot
    selectedOutcomes = sample(seqOutcome, round(Int64, proportion*Gens))
    # Construct vector of proportion ACCEPT beliefs (p in the model)
    propAccept = map(x -> x.modalBelief, selectedOutcomes) 
    # Concatenate into matrix (columns are series) 
    propAcceptMatrix = hcat(propAccept...)
        # Open plot named overall
    # Add sampled series with user-provided alpha

    overall = plot(propAcceptMatrix, label = :none, alpha = useralpha, linestyle = :dash, linecolor = :gray,
        title = "Average Modal Beliefs Trajectory")

    # Horizontal line at .5 for reference (.5 separates ACCEPT and REJECT beliefs)
    hline!(overall, [.5], linewidth = 2, color = :gray, alpha = 1.0, label = "") 
    # Add series for average beliefs overall (not sampled)
    plot!(overall, beliefsAverage.beliefAverage, linewidth = 5, color = :black, label = "Average Modal Beliefs") 

    ##### Conditional Plot (2)
    # Open plot named conditional
    conditional = plot(title = "Average Modal Beliefs Grouped by Outcome"); 

    # Plot sequence of marginal average accept beliefs for ACCEPT outcome
    plot!(conditional, beliefsAverage.Accepts_beliefAverage, color = :blue, linewidth = 5, label = "Ultimately Accepts") 
    # Plot sequence of marginal accept average beliefs for REJECT outcome 
    plot!(conditional, beliefsAverage.Rejects_beliefAverage, color = :red, linewidth = 5, label = "Ultimately Rejects") 
    # Plot horizontal line at .5 for reference
    hline!(conditional, [.5], linewidth = 2, color = :gray, label = "") 

    # Is proportion overall or of conditional distributions (to ensure you see some sample)
    if marginalproportion # If marginalproportion (user supplied), then the sampling of trajectories has already been taken
        selectedAccepts = filter(x -> x.majority > .5, selectedOutcomes) # Pull out accepts
        selectedRejects = filter(x -> x.majority < .5, selectedOutcomes) # Pull out accepts
        
    else # If not, then I need to condition on accepts/rejects
        selectedAccepts = sample(accepts, round(round(Int64, proportion*Gens)))
        selectedRejects = sample(rejects, round(round(Int64, proportion*Gens))) 
    end

    # Pull out beliefs
    # Accept beliefs for ACCEPT outcomes (should generally be > .5)
    propAccept_selectedAccepts = map(x -> x.modalBelief, selectedAccepts) 
    # Accept beliefs for REJECT outcomes (should generally be < .5)
    propAccept_selectedRejects = map(x -> x.modalBelief, selectedRejects) 
    propAcceptMatrix_Accepts = hcat(propAccept_selectedAccepts...) # Stick together into column matrix
    propAcceptMatrix_Rejects = hcat(propAccept_selectedRejects...) # Stick together into column matrix

    # Plot sample of conditioned trajectories
    plot!(conditional, propAcceptMatrix_Accepts, color = :blue, alpha = useralpha, linestyle = :dot, label = "") # ACCEPT distribution in blue, dotted
    plot!(conditional, propAcceptMatrix_Rejects, color = :red, alpha = useralpha, linestyle = :dash, label = "") # REJECT distribution in red, dashed
    
    # Construct AvgCondPlot (pane1)
    AvgCondPlot = plot(overall, conditional, layout=(2,1), size = (650, 700))

    ########### Threshold-Beliefs Plot

    df.OutcomeString .= ifelse.(df.FinalOutcome .== 1, "Ultimately Accepts", "Ultimately Rejects")
    df.OutcomeColor.= ifelse.(df.FinalOutcome .== 1, "blue", "red")

    pyplot()
    plot2 = groupedboxplot(df.SeqPosition, df.Threshold.-df.BeliefAccept, group = df.OutcomeString, color = [:red :blue], title = "Distribution of Threshold Beliefs - Actual Beliefs")
    
    # plot(beliefsAverage.Accepts_thresholdAverage, color = :blue, linestyle = :dash, linewidth = 3, label = "Average Threshold for Accepts")
    # plot!(beliefsAverage.Rejects_thresholdAverage, color = :red, linestyle = :dash, linewidth = 3, label = "Average Threshold for Rejects")
    # plot!(beliefsAverage.Accepts_beliefAverage, color = :blue, linewidth = 3, label = "Average Beliefs for Accepts") 
    # # Plot sequence of marginal accept average beliefs for REJECT outcome 
    # plot!(beliefsAverage.Rejects_beliefAverage, color = :red, linewidth = 3, label = "Average Beliefs for Rejects") 
    return (AvgCondPlot, plot2)
end
