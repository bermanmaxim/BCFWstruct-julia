

    using BCFWstruct


    function lossCB(param, y, ybar)
      abs(ybar - y)
    end
    
    function featureCB(param, x, y)
      [y,
       y * x,
       y * x^2,
       y * x^3,
       - 0.5 * y^2]
    end
    
    function oracleCB(param, model, x, y)  # loss-augmented inference
      w = model.w
      z = w[1] + w[2] * x + w[3] * x.^2 + w[4] * x.^3
        yhat = Float64[]
      if w[5] > 0
        yhat = [z - 1, z + 1] / w[5]
        yhat = max(min(yhat, 1),-1)
      end
        append!(yhat, [-1, 1])   
        augLoss = y_::Vector{Float64} -> abs(y_ - y) + z * y_ - 0.5 * y_.^2 * w[5]
      worse = indmax(augLoss(yhat))
        yhat = yhat[worse]
    end
    
    function oracleCB(param, model, x)    # inference
      w = model.w
        z = w[1] + w[2] * x + w[3] * x.^2 + w[4] * x.^3
        y = z / w[5]
    end




    oracleCB (generic function with 2 methods)




    x = collect(linspace(-pi,pi,21))
    y = 0.5*sin(x)
    y = y + 0.1*randn(size(y));


    xr = collect(linspace(minimum(x), maximum(x), 1024))
    yr = collect(linspace(-1,1,1024));


    param = BCFWstruct.Param(
        x,
        y,
        lossCB,
        oracleCB,
        featureCB
        )




    BCFWstruct.Param{Float64,Float64}([-3.14159,-2.82743,-2.51327,-2.19911,-1.88496,-1.5708,-1.25664,-0.942478,-0.628319,-0.314159  …  0.314159,0.628319,0.942478,1.25664,1.5708,1.88496,2.19911,2.51327,2.82743,3.14159],[-0.00853983,-0.309255,-0.271232,-0.342945,-0.526396,-0.510696,-0.54101,-0.425326,-0.237026,-0.0651087  …  0.330254,0.211964,0.243429,0.67993,0.528908,0.456445,0.554805,0.258425,0.074824,0.0439403],lossCB,oracleCB,featureCB)




    options = BCFWstruct.Options(Float64, Float64, 5, length(x))
    options.num_passes = 2000
    options.gap_threshold = 0.01;
    options.debug = true;


    model, progress = BCFWstruct.solverBCFW(param, options);

    running BCFW on 21 examples. The options are as follows:
    
    BCFWstruct.Options{Float64,Float64}(Float32[0.0f0,0.0f0,0.0f0,0.0f0,0.0f0],0x00000000000007d0,true,true,Inf32,true,0x0000000000000001,uniform::BCFWstruct.Sample,100,0.04761905f0,(Float64[],Float64[]),0.01,0x000000000000000a)
    pass 1 (iteration 21), SVM primal = 0.796107, SVM dual = 0.077083, duality gap = 0.719025, train_error = 0.305863 
    pass 2 (iteration 42), SVM primal = 0.826386, SVM dual = 0.099456, duality gap = 0.726930, train_error = 0.385318 
    pass 3 (iteration 63), SVM primal = 0.533878, SVM dual = 0.122905, duality gap = 0.410974, train_error = 0.125959 
    ... (output cut here) ...
    pass 1976 (iteration 41496), SVM primal = 0.452377, SVM dual = 0.438467, duality gap = 0.013910, train_error = 0.080122 
    pass 1977 (iteration 41517), SVM primal = 0.452376, SVM dual = 0.438477, duality gap = 0.013899, train_error = 0.080121 
    pass 2000 (iteration 42000), SVM primal = 0.452358, SVM dual = 0.438695, duality gap = 0.013663, train_error = 0.080112 
    Duality gap check: gap = 0.013663 at pass 2000 (iteration 42000)


Results:

    using PyPlot


    plot(progress.primal)
    plot(progress.dual);
    title("SSVM convergence");
    xlabel("iteration");


![png](images/output_9_0.png)



    w = model.w;
    z = w[1] + w[2] * xr + w[3] * xr.^2 + w[4] * xr.^3;
    F = yr*z' - 0.5 * yr.^2 * ones(size(z))' * w[5]; # scoring function
    F_ = F .- maximum(F, 1); # column rescaled
    y_fit = oracleCB(param, model, xr);


    imshow(F_, extent=[minimum(x), maximum(x), -1, 1], origin="lower")
    plot(xr, y_fit)
    plot(x, y, "o")
    xlim([minimum(x), maximum(x)]);


![png](images/output_11_0.png)



    
