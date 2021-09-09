"""
    快速排序步骤：1.选定基准pivot ;  2.根据pivot划分区间，左区间小于等于pivot，右区间大于pivot  ;  3.递归左右区间
    值得注意的是：针对递归，若想返回某一值，需要在每一递归分支加上return，否则总是为空None
"""
def partition(arr, low, high):
    begin = low - 1
    pivot = arr[high]
    for i in range(low, high):
        if arr[i] <= pivot:
            begin += 1
            arr[begin], arr[i] = arr[i], arr[begin]
    # 最后将pivot与begin的下一位交换
    arr[high], arr[begin+1] = arr[begin+1], arr[high]
    return begin+1


def quickSort(arr, low, high):
    if low < high:
        pivotIdx = partition(arr, low, high)
        quickSort(arr, low, pivotIdx-1)
        quickSort(arr, pivotIdx, high)
    return


if __name__ == '__main__':
    arr = [1, 3, 2, 5, 8]
    quickSort(arr, 0, 4)
    print(arr)
    a = {'1':1, "2": 2}
    print([ i for i in a.keys()])
