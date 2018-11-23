with open("2.txt",'r') as f:
    lines = f.readlines()
    nums = [int(line[:-1]) for line in lines]
with open("out2.txt", 'w') as f:
    f.write("id,classes\n")
    for i in range(228):
        f.write(str(i+1))
        f.write(',')
        f.write(str(nums[i]))
        f.write('\n')
