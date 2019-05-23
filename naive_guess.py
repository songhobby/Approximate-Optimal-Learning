prior = input("Prior_Favor: ")
x1=int(prior)/(int(prior)+4.)
x2=1/(int(prior)+4.)
x3=x1+x2
print("naive_guess_accuracy: {}".format(
        (4/64)*(0.5*x3+0.5*x1)+(24/64)*(0.25*x3+0.75*x1)+(36/64)*x1))