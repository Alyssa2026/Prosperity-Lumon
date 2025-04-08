

adj_lists = {
 'snowball': [('pizza', 1.45), ('silicon nugget', 0.52), ('seashell', 0.72)],
 'pizza': [('snowball', 0.7), ('silicon nugget', 0.31), ('seashell', 0.48)],
 'silicon nugget': [('snowball', 1.95), ('pizza', 3.1), ('seashell', 1.49)],
 'seashell': [('snowball', 1.34), ('pizza', 1.98), ('silicon nugget', 0.64)],
}

starting_item = "seashell"
final_item = "seashell"

curr_trades = []
max_change = 1
max_change_trades = []

def dfs(curr_item, curr_change, num_trades = 5):
    global max_change, max_change_trades
    if num_trades == 0:
        if curr_item == final_item and curr_change > max_change:
            max_change_trades = curr_trades.copy()
            max_change = curr_change
        return
    
    for next_item, change in adj_lists[curr_item]:
        curr_trades.append(next_item)
        dfs(next_item, curr_change * change, num_trades - 1)
        curr_trades.pop()
    
dfs(starting_item, 1)
print(max_change_trades)
print(max_change)