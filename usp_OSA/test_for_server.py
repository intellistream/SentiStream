import time

def timetest(t):
    return time.time()-t

if '__name__' == '__main__':
    print(timetest(3))
