from time import sleep
import sys

p = "|/\\"
p_len = len(p)

for i in range(21):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    # 20 => width of the string, -20 => append to end of string
    # sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.write("%s\b" % (p[i%p_len]))
    sys.stdout.flush()
    sleep(0.25)
