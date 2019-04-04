def model_search(data,test_fraction,random_state,cumulative_time,params,layers=None,restart=False):
    """
        docstring goes here, Rainie!
    """
    if restart is False and layers is None:
        raise Error("invalid parameter combination", "if restart is false, the number of hidden layers must be specified")
   
    X,Y,input_dim,output_dim = dp.data_info(data)
    proj_name = data['proj_name']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    
    if restart:
        y1 =pr.check_read("latest.json")
        starting_n = y1["starting_n"]
        print(y1)
        best_R = y1["best_R"]
        best_param = y1["best_param"]
        cumulative_time = y1["cumulative_time"]
        layers = y1["layer_number"]
        epoch_num = pr.get_epoch_num(layers)
        iterations = (len(units)*len(activation_functions))**(layers+1)*len(activation_functions)-starting_n
    else:
        iterations = (len(units)*len(activation_functions))**(layers+1)*len(activation_functions)

    inner_iterations = (len(units)*len(activation_functions))**layers
    options= make_combo(option1=activation_functions,option2=units)
    af_combs = make_pairwise_list(max_depth=layers, options=options)
    print(f'{layers}\t{options}\t{iterations} iterations required')
    iteration_n = 1

    if restart:
        run_once = 0
    else:
        for n in range(layers):
            best_param.append(['none','none'])
        run_once = 1

    
    for inner_iteration in range(inner_iterations):
        for option_in in options:
            inner_list=[]
            for k in range(layers):
                inner_list.append(af_combs[inner_iteration][k])
            for activation_out in activation_functions:
                if restart:
                    pass
                else:
                    print(inner_list)

                if run_once == 1:
                    print(f"running iteration {iteration_n}")
                else:
                    if restart and not (iteration_n > starting_n):
                        iteration_n += 1


                parameter_list = []
                parameter_list.append(option_in)
                parameter_list.extend(inner_list)
                parameter_list.append(activation_out)
                print(f"create input layer with activation of {option_in[0]} and units of {option_in[1]}")
                model = keras.Sequential()
                model.add(keras.layers.Dense(option_in[1],input_dim = input_dim,activation=option_in[0]))
                for i in range(len(inner_list)):
                    print(f"create hidden layer {i+1} of activation {inner_list[i][0]} and units {inner_list[i][1]}")
                    model.add(keras.layers.BatchNormalization(momentum=0.9))
                    model.add(keras.layers.Dense(inner_list[i][1],activation = inner_list[i][0]))
                print(f"create output layer with activation of {activation_out} and units of {output_dim}")
                model.add(keras.layers.Dense(output_dim,activation=activation_out))
                model.compile(loss='mean_squared_error', optimizer='adam',metrics=[R_squared])
                earlystop = keras.callbacks.EarlyStopping(monitor='val_R_squared',min_delta=0.0001,patience=20,mode='auto')
                
                if run_once is 0:
                    output_folder = './Results/%s_collection%d/intermediate_output%d' % (proj_name,layers,iteration_n)
                    file_ini = output_folder+'/weights-'+str(epoch_num)+'*'
                    filename = glob.glob(file_ini)
                    print(filename)
                    if os.path.isfile(filename[0]):
                        model.load_weights(filename[0])
                    else:
                        print("%s does not exists" % (filename[0]))

                collection_folder = './Results/%s_collection%d' % (proj_name,layers)
                if not os.path.exists(collection_folder):
                    os.makedirs(collection_folder)
                output_folder = './Results/%s_collection%d/intermediate_output%d'%(proj_name,layers,iteration_n)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                filepath=output_folder+"/weights-{epoch:02d}-{val_R_squared:.2f}.hdf5"
                checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_R_squared', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)
                callbacks_list = [earlystop,checkpoint]
                start = time.time()

                if run_once is 1:
                    history = model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
                else:
                    history = model.fit(X_train,Y_train,epochs=300,batch_size=10,callbacks=callbacks_list,validation_split=0.2,initial_epoch=epoch_num+1,verbose=0)


                end = time.time()
                cumulative_time += (end-start)
                print('it already took %0.2f seconds' % (cumulative_time))
                scores = model.evaluate(X_test,Y_test,verbose=0)
                if not os.path.exists("./Results/results%d.txt"%(layers)):
                    f = open("./Results/results%d.txt"%(layers),"w+")
                else:
                    f = open("./Results/results%d.txt"%(layers),"a+")
                f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                if scores[1]>best_R:
                    best_param = parameter_list
                    best_R = scores[1]
                else:
                    pass
                f.write("The best_R for now is %0.4f and combination is %s "% (best_R,best_param))

                run_once = 1

                iteration_n += 1
                x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                print(x)
                pr.check_write(x,'latest.json')
                print("")



    print(best_param)
    print(best_R)
    print('Training process has been finished')
    print('model took %0.2f seconds to train'%(cumulative_time))
    return best_param,best_R
