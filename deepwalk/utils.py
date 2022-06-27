def partition_num(num, workers):
    """获取能被当前workers 处理的任务数

    Args:
        num (_type_): _description_
        workers (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if num % workers == 0: #若能整除则均匀划分
        return [num//workers]*workers
    else: # 否则取其能被workers整除的最大值, 按前者均匀划分 , 再补充剩余部分
        return [num//workers]*workers + [num % workers]
