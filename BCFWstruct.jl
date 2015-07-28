module BCFWstruct

@enum Sample uniform perm

type Options{Tx,Ty}
    initial_w::Vector{Float64}
    num_passes::UInt
    do_line_search::Bool
    do_weighted_averaging::Bool
    time_budget::Float64
    debug::Bool
    rand_seed::UInt
    sample::Sample
    debug_multiplier::Int
    lambda::Float64
    test_data::Tuple{Vector{Tx}, Vector{Ty}}
    gap_threshold::Float6464
    gap_check::UInt
end


Options(Tx::Type, Ty::Type, d::Int, n::Int) = Options(
    zeros(Float64, d),     #initial_w
    UInt(200),          #num_passes
    true,          #do_line_search
    true,          #do_weighted_averaging
    Float64(Inf),          #time_budget
    false,          #debug
    UInt(1),          #rand_seed
    uniform,          #sample
    0,          #debug_multiplier
    Float64(1/n),          #lambda
    (Tx[],Ty[]),          #test_data
    0.1,          #gap_threshold
    UInt(10)          #gap_check
    )


type Param{Tx,Ty}
patterns::Vector{Tx}
labels::Vector{Ty}
lossFn::Function
oracleFn::Function
featureFn::Function
end

type Model
    w::Array{Float64}
    ell::Float64
end

type Progress
    primal::Vector{Float64}
    dual::Vector{Float64}
    eff_pass::Vector{Float64}
    train_error::Vector{Float64}
    test_error::Vector{Float64}  # only evaluated if options.test_data is given
end

Progress() = Progress([], [], [], [], [])


"""
 [model, progress] = solverBCFW(param, options)

 Solves the structured support vector machine (SVM) using the block-coordinate
 Frank-Wolfe algorithm, see (Lacoste-Julien, Jaggi, Schmidt, Pletscher; ICML
 2013) for more details.
 This is Algorithm 4 in the paper, and the code here follows a similar
 notation. Each step of the method calls the decoding oracle (i.e. the 
 loss-augmented predicion) only for a single point.

 The structured SVM has the form
 min_{w} 0.5*\lambda*||w||^2+ 1/n*\sum_{i=1}^n H_i(w) [see (3) in paper]
   where H_i(w) is the structured hinge loss on the i^th example:
         H_i(w) = max_{y in Y} L(y_i,y) - <w, \psi_i(y)> [(2) in paper]

 We use a calling interface very similar to version 1.1 of svm-struct-matlab 
 developped by Andrea Vedaldi (see vedaldi/code/svm-struct-matlab.html).
 svm-struct-matlab is a Matlab wrapper interface to the widely used
 SVM^struct code by Thorsten Joachims (http://svmlight.joachims.org/svm_struct.html) 
 which implements a cutting plane algorithm. See our ICML 2013 paper for
 experiments showing that the convergence per oracle call is much slower
 for SVM^struct than BCFW.
 
 If your code was using:
    model = svm_struct_learn(command_args,param)

 You can replace it with:
    model = solverBCFW(param, options)
 
 with the same param structure and letting options.lambda = 1/C
 to solve the *same* optimization problem. [Make sure your code only 
 uses model.w as we currently don't return the dual variables, etc. in the
 model structure unlike svm-struct-learn].
 
 Inputs:
   param: a structure describing the problem with the following fields:

     patterns  -- patterns (x_i)
         A cell array of patterns (x_i). The entries can have any
         nature (they can just be indexes of the actual data for
         example).
     
     labels    -- labels (y_i)
         A cell array of labels (y_i). The entries can have any nature.

     lossFn    -- loss function callback
         A handle to the loss function L(ytruth, ypredict) defined for 
         your problem. This function should have a signature of the form:
           scalar_output = loss(param, ytruth, ypredict) 
         It will be given an input ytruth, a ground truth label;
         ypredict, a prediction label; and param, the same structure 
         passed to solverBCFW.
 
     oracleFn  -- loss-augmented decoding callback
         [Can also be called constraintFn for backward compatibility with
          code using svm_struct_learn.]
         A handle to the 'maximization oracle' (equation (2) in paper) 
         which solves the loss-augmented decoding problem. This function
         should have a signature of the form:
           ypredict = decode(param, model, x, y)
         where x is an input pattern, y is its ground truth label,
         param is the input param structure to solverBCFW and model is the
         current model structure (the main field is model.w which contains
         the parameter vector).

     featureFn  feature map callback
         A handle to the feature map function \phi(x,y). This function
         should have a signature of the form:
           phi_vector = feature(param, x, y)
         where x is an input pattern, y is an input label, and param 
         is the usual input param structure. The output should be a vector 
         of *fixed* dimension d which is the same
         across all calls to the function. The parameter vector w will
         have the same dimension as this feature map. In our current
         implementation, w is sparse if phi_vector is sparse.
 
  options:    (an optional) structure with some of the following fields to
              customize the behavior of the optimization algorithm:
 
   lambda      The regularization constant (default: 1/n).

   gap_threshold **STOPPING CRITERION**
               Stop the algorithm once the duality gap falls below this
               threshold. Use gap_check to control how often the gap is
               checked. Note that the default of 0.1 is equivalent
               to the criterion used in svm_struct_learn Matlab wrapper.
               (default: 0.1).
               
   gap_check   Gives the number of passes through data between each check of the 
               duality gap (put 0 to turn it off). 
               I.e. every gap_check passes through the data,
               the algorithm do a batch pass through the data to compute
               the duality gap. [A feature to implement in the future is
               to do a batch FW step if the gap_threshold hasn't been met 
               yet to avoid wasting this computation.] This slows down the
               code roughly by a factor of (100/gap_check) %.
               (default: 10)

   num_passes  Maximum number of passes through the data before the 
               algorithm stops (default: 200)

   debug       Boolean flag whether to track the primal objective, dual
               objective, and training error (makes the code about 3x
               slower given the extra two passes through data).
               (default: 0)
   do_linesearch
               Boolean flag whether to use line-search. (default: 1)
   do_weighted_averaging
               Boolean flag whether to use weighted averaging of the iterates.
               *Recommended -- it made a big difference in test error in
               our experiments.*
               (default: 1)
   time_budget Number of minutes after which the algorithm should terminate.
               Useful if the solver is run on a cluster with some runtime
               limits. (default: inf)
   rand_seed   Optional seed value for the random number generator.
               (default: 1)
   sample      Sampling strategy for example index, either a random permutation
               ('perm') or uniform sampling ('uniform').
               [Note that our convergence rate proof only holds for uniform
               sampling, not for a random permutation.]
               (default: 'uniform')
   debug_multiplier
               If set to 0, the algorithm computes the objective after each full
               pass trough the data. If in (0,100) logging happens at a
               geometrically increasing sequence of iterates, thus allowing for
               within-iteration logging. The smaller the number, the more
               costly the computations will be!
               (default: 0)
   test_data   Struct with two fields: patterns and labels, which are cell
               arrays of the same form as the training data. If provided the
               logging will also evaluate the test error.
               (default: [])

 Outputs:
   model       model.w contains the parameters;
               model.ell contains b'*alpha which is useful to compute
               duality gap, etc.
   progress    Primal objective, duality gap etc as the algorithm progresses,
               can be used to visualize the convergence.

 Authors: Simon Lacoste-Julien, Martin Jaggi, Mark Schmidt, Patrick Pletscher
 Web: https://github.com/ppletscher/BCFWstruct

 Relevant Publication:
       S. Lacoste-Julien, M. Jaggi, M. Schmidt, P. Pletscher,
       Block-Coordinate Frank-Wolfe Optimization for Structural SVMs,
       International Conference on Machine Learning, 2013.
"""



function solverBCFW{Tx,Ty}(param::Param{Tx,Ty}, options::Union{Options{Tx,Ty}, Void})

# == getting the problem description:
phi = param.featureFn # for \phi(x,y) feature mapping
loss = param.lossFn # for L(ytruth, ypred) loss function
maxOracle = param.oracleFn # loss-augmented decoding function

patterns = param.patterns # {x_i} cell array
labels = param.labels # {y_i} cell array
n = length(patterns) # number of training examples

# general initializations
phi1 = phi(param, patterns[1], labels[1]) # use first example to determine dimension
d = length(phi1) # dimension of feature mapping

#if options is nothing, initialize to default
if options===nothing
    options = Options(Tx, Ty, d, n)
end

lambda = options.lambda

# === Initialization ===
# set w to zero vector
# (corresponds to setting all the mass of each dual variable block \alpha_(i)
# on the true label y_i coordinate [i.e. \alpha_i(y) =
# Kronecker-delta(y,y_i)] using notation from Appendix E of paper).

# w: d x 1: store the current parameter iterate
# wMat: d x n matrix, wMat[:,i] is storing w_i (in Alg. 4 notation) for example i.
#    Using implicit dual variable notation, we would have w_i = A \alpha_[i]
#    -- see section 5, "application to the Structural SVM"
ell = zero(Float64) # this is \ell in the paper. Here it is assumed that at the true label, the loss is zero
         # Implicitly, we have ell = b' \alpha

model = Model(zeros(Float64,d), 0.)
wMat = zeros(Float64, d,n) 

ellMat = zeros(Float64, n) # ellMat(i) is \ell_i in the paper (implicitly, ell_i = b' \alpha_[i])

if (options.do_weighted_averaging)
    wAvg = model.w # called \bar w in the paper -- contains weighted average of iterates
    lAvg = zero(Float64)
end

# logging
if (options.debug_multiplier == 0)
    debug_iter = n
    options.debug_multiplier = 100
else
    debug_iter = 1
end
progress = Progress()

println("running BCFW on $(length(patterns)) examples. The options are as follows:\n");
println(options)

srand(options.rand_seed)
time0 = time()


# === Main loop ====
k=0 # same k as in paper
gap_check_counter = 1 # keeps track of how many passes through the data since last duality gap check...
for p=1:options.num_passes

    perm = []
    if options.sample == perm
        perm = randperm(n)
    end

    for dummy = 1:n
        # (each numbered comment correspond to a line in algorithm 4)
        # 1) Picking random example:
        if options.sample == uniform
            i = rand(1:n) # uniform sampling
        else
            i = perm(dummy) # random permutation
        end
    
        # 2) solve the loss-augmented inference for point i
        ystar_i = maxOracle(param, model, patterns[i], labels[i])
                
        # 3) define the update quantities:
        # [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
        # psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
        psi_i =   phi(param, patterns[i], labels[i]) - phi(param, patterns[i], ystar_i)
        w_s = 1/(n*lambda) * psi_i
        loss_i = loss(param, labels[i], ystar_i)
        ell_s = 1/n*loss_i

        # sanity check, if this assertion fails, probably there is a bug in the
        # maxOracle or in the featuremap
        @assert (loss_i - dot(model.w, psi_i)) >= -1e-12
        
        # 4) get the step-size gamma:
        if (options.do_line_search)
            # analytic line-search for the best stepsize [by default]
            # formula from Alg. 4:
            # (lambda * (w_i-w_s)'*w - ell_i + ell_s)/(lambda ||w_i-w||^2)
            # equivalently, we get the following:
            gamma_opt = (dot(model.w,(wMat[:,i] - w_s)) - 1/lambda*(ellMat[i] - ell_s)) /
                        ( dot(wMat[:,i] - w_s, wMat[:,i] - w_s) +eps(Float64))
                     # +eps is to avoid division by zero...
            gamma = max(0.,min(1.,gamma_opt)) # truncate on [0,1]
        else
            # we use the fixed step-size schedule (as in Algorithm 3)
            gamma = 2*n/(k+2*n)
        end
        
        # 5-6-7-8) finally update the weights and ell variables
        model.w = model.w - wMat[:,i] # this is w^(k)-w_i^(k)
        wMat[:,i] = (1-gamma)*wMat[:,i] + gamma*w_s
        model.w = model.w + wMat[:,i] # this is w^(k+1) = w^(k)-w_i^(k)+w_i^(k+1)
        
        ell = ell - ellMat[i] # this is ell^(k)-ell_i^(k)
        ellMat[i] = (1-gamma)*ellMat[i] + gamma*ell_s
        ell = ell + ellMat[i] # this is ell^(k+1) = ell^(k)-ell_i^(k)+ell_i^(k+1)
    
        # 9) Optionally, update the weighted average:
        if (options.do_weighted_averaging)
            rho = 2/(k+2) # resuls in each iterate w^(k) weighted proportional to k
            wAvg = (1-rho)*wAvg + rho*model.w
            lAvg = (1-rho)*lAvg + rho*ell # this is required to compute statistics on wAvg -- such as the dual objective
        end
        
        k += 1
        
        # debug: compute objective and duality gap. do not use this flag for
        # timing the optimization, since it is very costly!
        # (makes the code about 3x slower given the additional 2 passes
        # through the data).
        if (options.debug && k >= debug_iter)
            model_debug = deepcopy(model)
            if (options.do_weighted_averaging)
                model_debug.w = wAvg
                model_debug.ell = lAvg
            else
                model_debug.w = model.w
                model_debug.ell = ell
            end
            f = -objective_function(model_debug.w, model_debug.ell, lambda) # dual value -equation (4)
            gap, w_s, ell_s = duality_gap(param, maxOracle, model_debug, lambda)
            primal = f+gap # a cheaper alternative to get the primal value
            train_error = average_loss(param, maxOracle, model_debug)
            @printf("pass %d (iteration %d), SVM primal = %f, SVM dual = %f, duality gap = %f, train_error = %f \n", 
                             p, k, primal, f, gap, train_error)

            push!(progress.primal, primal)
            push!(progress.dual, f)
            push!(progress.eff_pass, k/n)
            push!(progress.train_error, train_error)
            if (!isempty(options.test_data[1]))
                param_debug = Param(patterns_test, labels_test, loss, maxOracle, phi)
                test_error = average_loss(param_debug, maxOracle, model_debug)
                push!(progress.test_error, test_error)
            end

            debug_iter = min(debug_iter+n,ceil(Int, debug_iter*(1+options.debug_multiplier/100)))
        end

        # time-budget exceeded?
        t_elapsed = time() - time0
        if (t_elapsed/60 > options.time_budget)
            println("time budget exceeded.\n")
            if options.do_weighted_averaging
                model.w = wAvg # return the averaged version
                model.ell = lAvg
            else
                model.ell = ell
            end
            return
        end
    end
    # checking duality gap stopping criterion if required:
    if ((options.gap_check > 0) && (gap_check_counter >= options.gap_check))
        gap_check_counter = 0
        
        model_debug = deepcopy(model)

        # we use the wAvg parameter to compute gap
        if (options.do_weighted_averaging)
            model_debug.w = wAvg
            model_debug.ell = lAvg
        else
            model_debug.w = model.w
            model_debug.ell = ell
        end
        
        # compute gap:
        (gap, w_s, ell_s) = duality_gap(param, maxOracle, model_debug, lambda)
        # for later: [gap, w_s, ell_s] = duality_gap(param, maxOracle, model, lambda);
        if gap <= options.gap_threshold
            @printf("Duality gap below threshold -- stopping!\n")
            @printf("current gap: %f, gap_threshold: %f\n", gap, options.gap_threshold)
            @printf("Reached after pass: %d (iteration %d).\n", p, k)
            break # exit loop!
        else
            # to implement later: do a batch FW step with w_s & ell_s above
            @printf("Duality gap check: gap = %f at pass %d (iteration %d)\n", gap, p, k)
        end
    end # end of gap_check section
    gap_check_counter += 1
    
end # end of main loop

if (options.do_weighted_averaging)
    model.w = wAvg # return the averaged version
    model.ell = lAvg
else
    model.ell = ell
end

return (model, progress)

end # solverBCFW


# [ avg_loss ] = average_loss( param, maxOracle, model)
# Return the average loss for the predictions of model.w on
# input data param.patterns. See solverBCFW for interface of param.
# This function is expensive, as it requires a full decoding pass over all
# examples (so it costs as much as n BCFW iterations).

function average_loss( param, maxOracle, model )

    patterns = param.patterns
    labels = param.labels
    loss = param.lossFn

    loss_term = zero(Float64)
    for i=1:length(patterns)
        ystar_i = maxOracle(param, model, patterns[i]) # standard decoding (not loss-augmented) as no input label
        loss_term += loss(param, labels[i], ystar_i)
    end
    avg_loss = loss_term / length(patterns)
    
end


# [gap, w_s, ell_s] = duality_gap( param, maxOracle, model, lambda )
# 
# Return the SVM duality gap for the implicit primal-dual pair given by
#   model.w and model.ell (w = A*\alpha; ell = b'*\alpha -- alpha is
#   implicit). See "Duality Gap" in Section 4 of ICML paper.
#
# This function is expensive, as it requires a full decoding pass over all
# examples (so it costs as much as n BCFW iterations). If the duality gap
# is checked regularly as a stopping criterion, then one can also use the
# returned w_s & ell_s quantities to make a batch Frank-Wolfe step and not
# waste this computation (see the update in Alg. 2 in the paper).
#   
# duality gap = lambda*(w-w_s)'*w - ell + ell_s
#
# ell_s = 1/n \sum_i ell(y_i, ystar_i) -- the average loss for the
#          *loss-augmented* predictions ystar_i
#
# w_s = 1/(lambda*n) \psi_i(ystar_i)

function duality_gap( param, maxOracle, model, lambda )

    patterns = param.patterns
    labels = param.labels
    loss = param.lossFn
    phi = param.featureFn
    
    w = model.w
    ell = model.ell
    
    n = length(patterns)
    ystars = []
    for i=1:n
        # solve the loss-augmented inference for point i
        push!(ystars, maxOracle(param, model, patterns[i], labels[i]))
    end
    
    w_s = zeros(Float64, size(w))
    ell_s = zero(Float64) # note that we will normalize at the end to save rounding...
    for i=1:n
        w_s = w_s + (phi(param, patterns[i], labels[i])-phi(param, patterns[i], ystars[i]))
        ell_s = ell_s + loss(param, labels[i], ystars[i])
    end
      
    w_s = w_s/(lambda*n)
    ell_s = ell_s/n
    
    
    # computing duality gap:
    # note that this expression is also the numerator of the line-search quotient solution
    gap = lambda* dot(w, w - w_s) - ell + ell_s
   
    (gap, w_s, ell_s)
end

#OBJECTIVE_FUNCTION returns the function value f(alpha)
#   returns the SVM dual objective function  f(alpha) , which is
#   -equation (4) in the paper.
#   The arguments are w = A*alpha and b_alpha = b'*alpha.

function objective_function( w, b_alpha, lambda )
    lambda/2 * dot(w, w) - b_alpha
end


# [ avg_loss ] = average_loss(param, maxOracle, model)
#
# Computes the primal SVM objective value for a given model.w on
# input data param.patterns
# primal = lambda/2 * ||w||^2 + 1/n \sum_i max_y (L_i(y) - w*psi_i(y))
# hinge_losses is an n x 1 vector containing the terms in the sum over i
#
# This function is not used by our BCFW solver, but can be used to monitor
# progress of primal solvers, such a the srochastic subgradient method (SSG)
#
# This function is expensive, as it requires a full decoding pass over all
# examples (so it costs as much as n BCFW iterations).

function primal_objective( param, maxOracle, model, lambda )

    patterns = param.patterns
    labels = param.labels
    phi = param.featureFn
    loss = param.lossFn

    hinge_losses = zero(Float64)
    for i=1:length(patterns)
        # solve the loss-augmented inference for point i
        ystar_i = maxOracle(param, model, patterns[i], labels[i])
        
        loss_i = loss(param, labels[i], ystar_i)
        
        # feature map difference
        psi_i =   phi(param, patterns[i], labels[i]) - phi(param, patterns[i], ystar_i)
            
        # hinge loss for point i
        hinge_loss_i = loss_i - dot(model.w, psi_i)
        @assert hinge_loss_i >= 0.
        
        hinge_losses += hinge_loss_i
    end
    lambda/2*(model.w'*model.w) + hinge_losses/length(patterns)
    
end # primal_objective

end # end module BCFWstruct


BCFWstruct.solverBCFW(param) = BCFWstruct.solverBCFW(param, nothing)
