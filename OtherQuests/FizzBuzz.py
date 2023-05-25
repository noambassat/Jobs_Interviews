
dic = {3:"Fizz",5:"Buzz"}
for i in range(1,51):
    print_str = ""
    for k in dic.keys():
        if(i%k ==0): print_str +=dic[k]
    if(print_str==""): print(i)
    else: print(print_str)
