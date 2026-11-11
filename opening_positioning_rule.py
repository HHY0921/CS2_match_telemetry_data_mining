import pandas as pd
from sklearn.cluster import KMeans
from pureskillgg_dsdk import DsReaderFs, GameDsLoader
from pureskillgg_csgo_dsdk import add_player_vector_derived_columns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def preprocess_positions(match_data,global_k_means, target_seconds=30, time_window = 5, tick_rate=64, verbose = False):
    """
    提取防守方在回合開局(例如15秒時)的站位，並利用分群自動標註區域
    """
    df_vector = match_data['player_vector']
    df_round_start = match_data['round_start']

    # the round start at round 1, is this round 1 pistol round? IT IS
    # if we randomly add by 1 this can cause displacement
    # why round start at round 1 but vector start at round 2?????? STUPID
    # I think we either remove pistol rounds, or decrease vector by 1
    # df_round_start = df_round_start.drop(df_round_start[df_round_start['round'] == 1].index)
    df_vector['round'] = df_vector['round'] - 1 

    round_starts = df_round_start[['round', 'tick']].rename(columns={'tick': 'start_tick'})
    
    # calculate target_tick
    round_starts['target_tick'] = round_starts['start_tick'] + (target_seconds * tick_rate)

    if verbose:
        print("\n   ======== round_start stats ========\n")
        print(round_starts[['round', 'start_tick', 'target_tick']].head())
    
    df_merged = pd.merge(df_vector, round_starts, on='round', how='inner')
    
    df_ct_early = df_merged[df_merged['team_code'] == 3].copy() # 3 = CT
    # # there's a 15 second idle time, shop time??
    if verbose:
        print("\n   ======== player vector merged with ticks(CT side) ========\n")
        print(df_ct_early[['round', 'player_id', 'start_tick']].head(10))
    
    # get average position on (target second - time_window)
    window_start = df_ct_early['start_tick'] + ((target_seconds - time_window) * tick_rate)
    window_end = df_ct_early['start_tick'] + (target_seconds * tick_rate)
    df_window = df_ct_early[(df_ct_early['tick'] >= window_start) & (df_ct_early['tick'] <= window_end)]
    df_ct_positions = df_window.groupby(['round', 'player_id'])[['x_pos', 'y_pos', 'z_pos']].mean().reset_index()

        
        #  df_ct_early[df_ct_early['player_id'] == 65280]
        # df_round_2 = df_ct_early[df_ct_early['round'] == 2].copy() # there's no round 1
        # print(df_window.head())
        # print(df_ct_positions.head())

    # round 2: 6037 - 10662, delta = 4625 (72)
    # round 3: 10665 - 19430, delta = 8765 (136)
    # round 4: 19433 - 26879, delta = 7466
    # round 5: 26881 - 33015, delta = 6204

    # 4. 利用 K-Means 將座標分為 "n_clusters" 個防守區域，給予每個座標一個區域標籤 (Cluster_0 ~ Cluster_n)
    # K=7: A platform, A tunnel, A arc, CT spawn/CT door, B bomb, B tunnel, A tunnel close to T side
    kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
    df_ct_positions['zone_label'] = global_k_means.predict(df_ct_positions[['x_pos', 'y_pos', 'z_pos']]) # train
    df_ct_positions['zone_label'] = 'Zone_' + df_ct_positions['zone_label'].astype(str)

    if verbose:
        print("\n   ======== time relationship with movement ========\n")
        print(df_ct_positions[['round', 'player_id', 'zone_label']].head(10))
        plot_defense_zones(df_ct_positions.copy())
        
    return df_ct_positions


# for weapon, do we make all rifles just "rifle" and SMG just "SMG"
# this way association can be simpler, but some sick gun like AK will lose its info
def extract_first_blood(match_data, df_ct_position, verbose=False):
    """提取每回合的第一個擊殺，並判斷是否為 CT 拿下"""
    # df_death start at round 2 -> no pistol kills
    df_death = match_data['player_death']

    df_death_sorted = df_death.sort_values(by=['round', 'tick'])
    first_bloods = df_death_sorted.groupby('round').first().reset_index()

    first_bloods['CT_Got_First_Blood'] = first_bloods['attacker_team_code'] == 3
    first_bloods['CT_Got_First_Blood'] = first_bloods['CT_Got_First_Blood'].map({True: 'FB_CT', False: 'FB_T'})

    
    df_ct_position_attacker = df_ct_position.rename(columns={'player_id': 'attacker_id'})
    # print(df_ct_position_attacker.head(10))
    first_bloods_by_zone = pd.merge(first_bloods, df_ct_position_attacker, how="left", on=['round', 'attacker_id'])

    if verbose:
        print("\n   ======== FB info ========\n")
        print(first_bloods[['round', 'tick', 'player_id', 'attacker_id', 'CT_Got_First_Blood', 'weapon_name']].head(5))
        print("\n   ======== FB by zone info ========\n")
        print(first_bloods_by_zone[['round', 'tick', 'attacker_id', 'attacker_team_code', 'weapon_name','CT_Got_First_Blood', 'zone_label']].head(20))

    return first_bloods_by_zone[['round', 'tick', 'attacker_id', 'attacker_team_code', 'weapon_name','CT_Got_First_Blood', 'zone_label']]

def extract_CT_weapon(match_data, df_ct_positions, target_seconds=20, tick_rate=64, verbose=False):  
    df_equip = match_data['item_equip']
    df_round_start = match_data['round_start']
    # print(df_equip[['round', 'tick', 'item']].head(20))
    round_starts = df_round_start[['round', 'tick']].rename(columns={'tick': 'start_tick'})

    df_merged = pd.merge(df_equip, round_starts, on='round')    

    is_early = df_merged[df_merged['tick'] <= df_merged['start_tick'] + (target_seconds * tick_rate)]
    is_ct = is_early[is_early['player_team_code'] == 3]
    # delete small weapons
    exclude_list = ['knife', 'c4', 'flashbang', 'smokegrenade', 'hegrenade', 'molotov', 'incgrenade', 'decoy', 'taser', 
                    'glock', 'hkp2000', 'usp_silencer', 'p250', 'tec9', 'fiveseven', 'cz75a', 'deagle', 'revolver']
    is_primary = is_ct[~is_ct['item'].isin(exclude_list)].copy()
    # print(is_primary.head(10))

    last_equip = is_primary.sort_values(by=['round', 'tick']).groupby(['round', 'player_id']).last().reset_index()

    weap_pos = pd.merge(last_equip, df_ct_positions[['round', 'player_id', 'zone_label']], on=['round', 'player_id'], how='inner')
    # print(weap_pos.head(10))
    
    # 組合 Item 字串，例如 "Zone_3_Hold_awp"
    weap_pos['zone_weapon_item'] = weap_pos['zone_label'] + "_Hold_" + weap_pos['item']
    
    if verbose:
        print("\n   ======== Early Weapons Info ========\n")
        print(weap_pos[['round', 'player_id', 'zone_weapon_item']].head(10))
        
    return weap_pos[['round', 'player_id', 'zone_weapon_item']]

def extract_early_utility(match_data, df_ct_positions, target_seconds=20, tick_rate=64, verbose=False):
    """提取 CT 在開局 20 秒內是否有丟擲關鍵道具 (煙霧彈、閃光彈)"""
    df_fire = match_data['weapon_fire']
    df_round_start = match_data['round_start']
    ALL_GRENADE_NAME = ['flashbang', 'hegrenade', 'smokegrenade', 'molotov', 'incgrenade','decoy']

    round_starts = df_round_start[['round', 'tick']].rename(columns={'tick': 'start_tick'})
    df_fire_merged = pd.merge(df_fire, round_starts, on='round')
    
    # 條件 1: 開局 20 秒內
    is_early = df_fire_merged[df_fire_merged['tick'] <= df_fire_merged['start_tick'] + (target_seconds * tick_rate)]
    # 條件 2: 是 CT 丟的
    is_ct = is_early[is_early['player_team_code'] == 3]
    # 條件 3: 是道具 
    is_grenade = is_ct.loc[is_ct['weapon_name'].isin(ALL_GRENADE_NAME)]

    util_pos = pd.merge(is_grenade, df_ct_positions[['round', 'player_id', 'zone_label']], on=['round', 'player_id'], how='inner')
    util_pos['zone_utility_item'] = util_pos['zone_label'] + "_Threw_" + util_pos['weapon_name']

    if verbose:
        print("\n   ======== early fire info after filter ========\n")
        print(is_grenade[['round', 'tick','player_team_code','weapon_name']].head(10))

        print("\n ======== Early Utility Info ========\n")
        print(util_pos[['round', 'player_id', 'zone_utility_item']].head(10))

    return util_pos[['round', 'player_id', 'zone_utility_item']]

def build_association_transactions(df_positions, df_first_blood, df_weapons, df_utilities):
    """
    將所有特徵整併成 Apriori 演算法接受的交易格式 (Transactions)
    一行代表一回合，內容為各項特徵的標籤清單
    """
    transactions = []
    rounds = df_positions['round'].unique()
    
    for r in rounds:
        transaction = []
        
        # 1. 放入該回合 CT 佔據的防區 (如 Zone_0, Zone_1)
        zones = df_positions[df_positions['round'] == r]['zone_label'].unique()
        transaction.extend(zones)
        
        # 2. 放入防區武器配置 (如 Zone_0_Hold_awp)
        weapons = df_weapons[df_weapons['round'] == r]['zone_weapon_item'].unique()
        transaction.extend(weapons)
        print(weapons)
        # transaction.extend()
        
        # 3. 放入防區道具丟擲 (如 Zone_1_Threw_smokegrenade)
        utils = df_utilities[df_utilities['round'] == r]['zone_utility_item'].unique()
        transaction.extend(utils)
        
        # 4. 放入首殺結果 (多粒度輸出)
        fb_round = df_first_blood[df_first_blood['round'] == r]
        if not fb_round.empty:
            fb_data = fb_round.iloc[0]
            
            if fb_data['CT_Got_First_Blood'] == 'FB_CT':
                transaction.append('FB_CT') # 最宏觀指標
                
                # 附加粒度更細的規則 (武器與防區)
                if pd.notna(fb_data['weapon_name']):
                    transaction.append(f"FB_CT_With_{fb_data['weapon_name']}")
                if pd.notna(fb_data['zone_label']):
                    transaction.append(f"FB_CT_From_{fb_data['zone_label']}")
            else:
                transaction.append('FB_T')
                
        transactions.append(transaction)
        
    return transactions

def plot_defense_zones(df_positions):
    """
    將 K-Means 分群後的 CT 座標繪製成 2D 散佈圖
    """
    plt.figure(figsize=(10, 10))
    
    # 取得所有的分群標籤 (Zone_0 ~ Zone_4)
    zones = df_positions['zone_label'].unique()
    
    # 為每個區域設定不同顏色
    colors = plt.get_cmap('tab10', len(zones))
    
    for i, zone in enumerate(zones):
        # 篩選出該區域的資料點
        subset = df_positions[df_positions['zone_label'] == zone]
        
        # 畫在圖上
        plt.scatter(
            subset['x_pos'], 
            subset['y_pos'], 
            label=zone, 
            color=colors(i), 
            alpha=0.6,      # 透明度，方便看疊加的密度
            edgecolors='white',
            linewidth=0.5,
            s=50            # 點的大小
        )
        
    plt.title('CS2 Dust2 CT Opening Positions (K-Means Clustering)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # 將背景設定為深色，讓座標點更明顯 (類似遊戲引擎視角)
    plt.gca().set_facecolor('#2b2b2b') 
    
    plt.legend(title="Defense Zones")
    plt.grid(True, alpha=0.2, color='white')
    plt.show()

# 呼叫方式：
# plot_defense_zones(df_ct_positions)

# --- 執行流程 ---
# df_positions = preprocess_positions(match_data)
# df_fb = extract_first_blood(match_data)
# smoke_rounds = extract_early_utility(match_data)
# transactions = build_association_transactions(df_positions, df_fb, smoke_rounds)
# print(transactions[0]) 
# 輸出預期: ['Zone_1', 'Zone_4', 'Used_Early_Smoke', 'FB_CT']
# output v2: ['Zone_0', 'Zone_1', 'Zone_0_Hold_AWP', 'Zone_1_Hold_M4A4', 'Zone_1_Threw_smokegrenade', 'FB_CT', 'FB_CT_Zone_0']

# do we need attack information for first blood (eg. killed in Zone 1, using AWP) and put them as an item or multiple items?
# -> seperate like this: 
# 1. zone and weapon like "Zone_2_AWP"
# 2. "FB_CT", "FB_CT_AWP", "FB_CT_Zone_2"
# do we need all thrown utilities info? -> does each grenade count as an item in association
# -> yes, mand merge like "Zone_1_Threw_smokegrenade"

def run_apriori(transactions_list, min_support=0.05, min_confidence=0.4):

    print("Start Apriori")
    te = TransactionEncoder()
    te_ary = te.fit(transactions_list).transform(transactions_list)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

    # min_support 0.05 代表該戰術在所有回合中至少要出現 5% 才會被納入
    frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True, max_len=5)

    if frequent_itemsets.empty:
        print("找不到符合最低 Support 門檻的頻繁項目集，請嘗試調低 min_support。")
        return
        
    # 3. 產生關聯規則 (Association Rules)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    def is_target_consequent(consequents):
        # 檢查 frozen set 裡面是否有任何字串以 FB_CT 開頭
        return any(str(item).startswith('FB_CT') for item in consequents)
        
    # target_rules = rules[rules['consequents'].apply(is_target_consequent)].copy()
    target_rules = rules[rules['consequents'] == frozenset({'FB_CT'})].copy()

    if target_rules.empty:
        print("找不到指向 CT 首殺的有效規則，請嘗試調低 min_confidence。")
        return

    target_rules = target_rules.sort_values(by='lift', ascending=False)
    
    print(f"成功萃取出 {len(target_rules)} 條戰術規則！\n")
    print("--- 🏆 Top 10 最強開局戰術 (依 Lift 排序) ---\n")
    
    for index, row in target_rules.head(10).iterrows():
        ant = ", ".join(list(row['antecedents']))
        con = ", ".join(list(row['consequents']))
        
        print(f"戰術條件: [{ant}] => 預期結果: [{con}]")
        print(f"   ▶ Support    (發生頻率): {row['support']:.3f}")
        print(f"   ▶ Confidence (戰術成功率): {row['confidence']:.3f}")
        print(f"   ▶ Lift       (效果提升倍數): {row['lift']:.3f}\n")
        
    return target_rules

def start_analyze(match_data,global_k_means,first_match, verbose=False):

    # for target_sec in range(15, 20):
    df_positions = preprocess_positions(match_data,global_k_means,target_seconds=28, time_window=3, verbose=verbose)
    if first_match:
        plot_defense_zones(df_positions=df_positions.copy())
    if verbose:
            print("\n======== CLUSTER POSITION DONE ========\n")
    df_fb = extract_first_blood(match_data, df_positions, verbose=verbose)
    df_weapon = extract_CT_weapon(match_data, df_positions, verbose=verbose)
    df_early_util = extract_early_utility(match_data, df_positions ,target_seconds=28, verbose=verbose)
    if verbose:
        print("\n======== EXTRACT First Blood/Early Utility DONE ========\n")
    transactions = build_association_transactions(df_positions, df_fb, df_weapon, df_early_util)
    if verbose:
        print("\n======== TRANSACTIONALIZE DONE ========\n")
        print(len(transactions))
    # for transaction in transactions:
    #     if 'FB_CT' in transaction:
    #         print(transaction)
    return transactions
