(* pure pascal neural network*)

(*/////////////////////////*)
(*    author: grafstor*)
(*    date: 27.06.20*)
(*/////////////////////////*)

(*verison 1.0*)

program Hello;
uses Crt;

var 
    train_x: array [1..12] of integer;
    train_y: array [1..4] of real;
    weigths: array [1..4] of real;
    dot_product: array [1..4] of real;
    output: array [1..4] of real;
    error: array [1..4] of real;
    sigmoid_derivative: array [1..4] of real;
    delta: array [1..4] of real;
    adjustment: array [1..3] of real;
    train_x_T: array [1..12] of integer;
    
    test: array [1..3] of real;
    
    summary, result: real;
    
    i, j, p, epoch: integer;

begin

    weigths[0] := -0.145;
    weigths[1] := -0.993;
    weigths[2] := -0.359;

    train_y[0] := 0;
    train_y[1] := 1;
    train_y[2] := 1;
    train_y[3] := 0;
    
    
    train_x[0] := 0;
    train_x[1] := 0;
    train_x[2] := 1;
    
    train_x[3] := 1;
    train_x[4] := 1;
    train_x[5] := 1;
    
    train_x[6] := 1;
    train_x[7] := 0;
    train_x[8] := 1;
    
    train_x[9] := 0;
    train_x[10] := 1;
    train_x[11] := 1;


    train_x_T[0] := 0;
    train_x_T[1] := 1;
    train_x_T[2] := 1;
    train_x_T[3] := 0;
    
    train_x_T[4] := 0;
    train_x_T[5] := 1;
    train_x_T[6] := 0;
    train_x_T[7] := 1;
    
    train_x_T[8] := 1;
    train_x_T[9] := 1;
    train_x_T[10] := 1;
    train_x_T[11] := 1;

    
    for  epoch := 0 to 1000 do
    begin
        
        for  i := 0 to 3 do
        begin
            summary := 0;
        
            for p := 0 to 2 do
                summary := summary + train_x[i*3+p] * weigths[p];

            dot_product[i] := summary;
        end;
        
        for i := 0 to 3 do
            output[i] := 1 / (1 + exp(-dot_product[i]*LN(2.71828182846)));

        
        
        for i := 0 to 3 do
            error[i] := train_y[i] - output[i];
        
    
    
        for i := 0 to 3 do
            sigmoid_derivative[i] := output[i] * (1 - output[i]);

        
        for i := 0 to 3 do
            delta[i] := error[i] * sigmoid_derivative[i];

        
    
        for  i := 0 to 2 do
        begin
            summary := 0;
        
            for p := 0 to 3 do
                summary := summary + train_x_T[i*4+p] * delta[p];

            adjustment[i] := summary;
        end;
        
        
        for i := 0 to 2 do
            weigths[i] := weigths[i] + adjustment[i];
    end;
    

    test[0] := 1;
    test[1] := 0;
    test[2] := 1;
    
    (* test predict *)
    summary := 0;
        
    for p := 0 to 2 do
        summary := summary + test[p] * weigths[p];
    
    result := 1 / (1 + exp(-summary*LN(2.71828182846)));

    write('test result: ');
    writeln(result);
    
    
    writeln('weigths: ');
    for i := 0 to 2 do
        writeln(weigths[i]);
end.

