#-------------------------------#
#       Author: grafstor        
#       Date: 08.06.20          
#-------------------------------#

from geras import *

def main():
    x = np.array([[1,0,1],
                  [0,0,1],
                  [1,1,0],
                  [0,1,0]])

    y = np.array([[1,
                   0,
                   1,
                   0]]).T

    model = Model(

       Dense(3),
       Sigmoid(),

       Dense(1),
       Sigmoid(),

    )

    model.fit(x, y, 10000)

    print(model.predict([[1,0,1]]))
    print(model.predict([[0,0,1]]))

if __name__ == '__main__':
    main()
