def search(nums, target):
    mid = int(len(nums) / 2)
    if (len(nums) == 2):
        for z in nums:
            if (z == target):
                return nums.index(z)
    if (mid > target):
        for i in nums[mid::1]:
            if (i == target):
                return nums.index(i)
    if (mid < target):
        for j in nums[mid::-1]:
            if (j == target):
                return nums.index(j)
    return -1

nums = [1,3,5]
print (search(nums,0))