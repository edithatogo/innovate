using Innovate
println(kernel_predict_model("logistic", Dict("inputs"=>Dict("time"=>[0.,1.,2.]), "parameters"=>Dict("L"=>100.,"k"=>1.,"x0"=>1.))))
