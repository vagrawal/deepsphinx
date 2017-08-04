import sys

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: {} <sym>".format(__file__))
        sys.exit(1)
    sym_id = len(open(sys.argv[1]).readlines())
    with open(sys.argv[1], 'a') as f:
        f.write('<backoff> ' + str(sym_id))
    while True:
        try:
            inp = raw_input().split()
            # 0 = epsilon symbol id
            if (len(inp) > 3 and inp[2] == '0' and inp[3] == '0'):
                inp[2] = str(sym_id)
            print '\t'.join(inp)
        except:
            break

