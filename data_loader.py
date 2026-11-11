import glob
import os
import pandas as pd
import numpy as np
import pickle

from pureskillgg_dsdk import DsReaderFs, GameDsLoader
from pureskillgg_csgo_dsdk import add_player_vector_derived_columns

ROOT_DIR = "data"


# Init Loader (eg. keys[0] means first match in the list)
# Each match is broken into 30+ "channels" each channel corresponds to either some event or is telemetry data.
# more than 1 in key part retunrs a list

# keys = sorted(
#     os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")
#     for path in glob.glob(os.path.join(ROOT_DIR, "csds", "*", "*", "*", "*", "csds"))
# )
# print(f"總共找到了 {len(keys)} 場比賽！")
# cnt = 0
# df_dust = pd.DataFrame()
# for i in range(0, 15):
#     loader = GameDsLoader(reader=DsReaderFs(root_path=ROOT_DIR, manifest_key=keys[i]))
#     match_data = loader.get_channel({"channel": "header"})
#     if match_data["map_name"][0] == "de_dust2":
#         cnt += 1
#         df_dust = pd.concat([df_dust, match_data[["map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]]])

# print(cnt)
# print(df_dust[["map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]].head())

key_mask = []

def get_mask_with_map_name(map: str):
    
    keys = sorted(
        os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")
        for path in glob.glob(os.path.join(ROOT_DIR, "csds", "*", "*", "*", "*", "csds"))
    )
    print(f"總共找到了 {len(keys)} 場比賽！")
    cnt = 0
    match_list = []
    for i in range(0, 1351):
        loader = GameDsLoader(reader=DsReaderFs(root_path=ROOT_DIR, manifest_key=keys[i]))
        match_data = loader.get_channel({"channel": "header"})
        if match_data["map_name"][0] == map:
            key_mask.append(keys[i])
            cnt+=1
            
    print(key_mask)
    print(f"Total {cnt} matches found for map: {map}")
    return 

def get_match_with_key_mask(key_mask_list: list, max_matches = 3):
    
    keys = sorted(
        os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")
        for path in glob.glob(os.path.join(ROOT_DIR, "csds", "*", "*", "*", "*", "csds"))
    )
    print(f"總共找到了 {len(keys)} 場比賽！")
    cnt = 0
    match_list = []
    for i in range(50, 1351):
        if cnt >= max_matches:
            break
        if keys[i] in key_mask_list:
            loader = GameDsLoader(reader=DsReaderFs(root_path=ROOT_DIR, manifest_key=keys[i]))
            match_data = loader.get_channels([
                {"channel": "header"},
                {"channel": "player_vector"},
                {"channel": "round_start"},
                {"channel": "player_death"},
                {"channel": "weapon_fire"},
                {"channel": "item_equip"}
            ])
            cnt += 1
            yield match_data

    print(f"Total {cnt} matches found for the given key mask.")
# dust 2 alone have like 16GB of file -> cannot store


# Most channel listed:
# Header: header
# Player info: player_info, player_personal
# Telemetry(Something on 1 tick):  
#   player_vector, player_status, tick 
#   grenade_vector, world_item_vector, molotov_fire
#   player_inputs, player_command, player_subtick_input
# Single Event: player_death, player_hurt, weapon_fire, round_end
# Multi-event: bomb_action, bomb_state, grenade_state, molotov_state, round_state

# round start at round 2

# more on https://docs.pureskill.gg/datascience/adx/cs2/csds/spec

# ['bomb_action', 'bomb_defuse', 'bomb_state', 'bot_takeover', 'grenade_state', 'header',
#  'item_equip', 'item_pickup', 'molotov_state', 'other_death', 'player_action', 'player_blind',
#  'player_death', 'player_disconnect', 'player_footstep', 'player_hurt', 'player_info', 'player_name',
#  'player_personal', 'player_spawn', 'player_status', 'player_vector', 'round_end', 'round_mvp', 'round_start',
#  'round_state', 'tick', 'weapon_action', 'weapon_fire', 'item_remove']

# ['bomb_action', 'bomb_defuse', 'bomb_state', 'bot_takeover', 'grenade_state', 'header',
#  'item_equip', 'item_pickup', 'molotov_state', 'other_death', 'player_action', 'player_blind',
#  'player_death', 'player_disconnect', 'player_footstep', 'player_hurt', 'player_info', 'player_name',
#  'player_personal', 'player_spawn', 'player_status', 'player_vector', 'round_end', 'round_mvp', 'round_start',
#  'round_state', 'tick', 'weapon_action', 'weapon_fire', 'item_remove']




# def tmp():
#     bigbigdata = loader.get_channels()

#     print(bigbigdata.keys())

#     match_data = loader.get_channels([
#         {"channel": "header"}, 
#         {"channel": "player_death"}
#     ])
#     # 讀取玩家每一幀的動態資料
#     df_vector = loader.get_channel({"channel": "player_vector"})
#     df_round_end = loader.get_channel({"channel": "round_end"})

#     df_header = match_data["header"]
#     df_death = match_data["player_death"]

#     print("\n-------- match header --------")
#     print(df_header[["map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]].head())
#     print(df_header.columns.values)
#     print("Number of rows:" ,df_header.shape[0])

#     print("\n--- death info ---")
#     print(df_death[["round", "tick", "weapon_name", "is_headshot"]].head())
#     print("Number of death:",df_death.shape[0])

#     print("\n--- score update ---")
#     # print(df_round_end[["round", "winner_team_code", "win_reason_code", "player_count"]].head(20))
#     # print(df_round_end[["round", "team_code", "old_score", "new_score"]].head(20))

#     # player info should have 10 each round
#     print("\n--- player info ---")
#     # print(df_player_info.columns.values)
#     # print(df_player_info[["round", "player_id"]].head(20))
#     # print(df_player_info.shape[0])


#     # 自動計算出移動速度、視角角度等進階欄位 (直接附加在原本的 dataframe 上)
#     add_player_vector_derived_columns(df_vector)

#     # 查看計算出來的 X, Y 軸速度
#     # print("\n--- 移動Vector速度) ---")
#     # print(df_vector[df_vector["x_vel"] != 0].head())



