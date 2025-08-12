# -*- coding: utf-8 -*-

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=10):
    """
    Early stopping strategy to prevent overfitting.
    防止过拟合的早停策略。

    Args:
        log_value (float): The validation metric value of the current epoch. / 当前周期的验证指标值。
        best_value (float): The best validation metric value observed so far. / 目前为止观察到的最佳验证指标值。
        stopping_step (int): The number of consecutive epochs without improvement. / 连续未改善的周期数。
        expected_order (str, optional): 'acc' for accuracy-like metrics (higher is better), 
                                     'dec' for loss-like metrics (lower is better). 
                                     Defaults to 'acc'.
                                     'acc'表示精度类指标（越高越好），'dec'表示损失类指标（越低越好）。
                                     默认为'acc'。
        flag_step (int, optional): The patience, i.e., the number of epochs to wait for 
                                 improvement before stopping. Defaults to 10.
                                 耐心值，即在停止前等待改善的周期数。默认为10。

    Returns:
        tuple: A tuple containing the updated best value, the updated stopping step, 
               and a boolean indicating whether to stop training.
               一个元组，包含更新后的最佳值、更新后的停止步数和一个指示是否停止训练的布尔值。
    """
    if (expected_order == 'acc' and log_value >= best_value) or \
       (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    should_stop = stopping_step >= flag_step
    
    return best_value, stopping_step, should_stop
