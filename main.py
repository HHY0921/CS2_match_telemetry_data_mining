import data_loader
import opening_positioning_rule
import pandas as pd
from sklearn.cluster import KMeans
import random
KEY_MASK_DUST2 = ['csds/2023/12/10/0826b896-52ee-4bb5-9d00-165e599fdaee/csds', 'csds/2023/12/10/102860e5-87ed-4b47-a129-27ba0b39267f/csds', 'csds/2023/12/10/12290dd9-63bd-41a5-8408-b03fe72af2c4/csds', 'csds/2023/12/10/15b65468-651a-4ce3-b3e5-7755623f3b4d/csds', 'csds/2023/12/10/23aaf423-615d-45c4-82a0-837a782d831f/csds', 'csds/2023/12/10/349ef7c0-77eb-4777-9d4a-80d8d2d65e6c/csds', 'csds/2023/12/10/36277387-28ba-4706-bc2a-fb2ba6da5365/csds', 'csds/2023/12/10/3a3e27ea-676f-4959-9599-8b7d11a25763/csds', 'csds/2023/12/10/4b2a7011-e466-48fa-a117-4870cb7a1910/csds', 'csds/2023/12/10/4e0cde01-f4b6-4d69-a16d-16c22dceb48d/csds', 'csds/2023/12/10/579acf3c-d6a8-4a89-8de6-594f7ecfc74c/csds', 'csds/2023/12/10/58454819-c5a9-4b2d-afcb-9f800b5775f0/csds', 'csds/2023/12/10/5a8f1a0f-732c-47a1-a8e1-cd87c21595c3/csds', 'csds/2023/12/10/5ec8e531-e640-4dd7-9fa7-fbb3510fa155/csds', 'csds/2023/12/10/5f8c4089-5ce1-47cc-9d77-053563e4609e/csds', 'csds/2023/12/10/63445000-21ea-4f5e-84ac-f19fa8d81743/csds', 'csds/2023/12/10/7315b42b-397f-4ef3-afcd-6369e8a29249/csds', 'csds/2023/12/10/752f18b4-eeba-4825-9b15-b7bba050bc0f/csds', 'csds/2023/12/10/756c0d58-0b19-485c-bac8-da4de590679f/csds', 'csds/2023/12/10/7d0f0ee0-f7c9-4944-bfdb-41f99865ed68/csds', 'csds/2023/12/10/8289b516-8104-4822-8aed-e3fef5936680/csds', 'csds/2023/12/10/87b66b98-be62-496c-9ec4-e96eacb5b233/csds', 'csds/2023/12/10/90a645c3-6586-43b6-9a7a-ec7094696efb/csds', 'csds/2023/12/10/9205df51-9a36-42ba-be39-ba5d36a3d41c/csds', 'csds/2023/12/10/9601f269-d14f-4ad0-b230-844e44963c99/csds', 'csds/2023/12/10/a1d546f8-4830-4711-b522-8697544d6ba8/csds', 'csds/2023/12/10/a52206c8-a339-461a-b3d6-f4a9b8ec49c5/csds', 'csds/2023/12/10/a57abe4c-51b9-4f61-88b5-077777b7572d/csds', 'csds/2023/12/10/ad21e4f1-3d82-4700-be0f-df29ce7097bf/csds', 'csds/2023/12/10/aeab3415-e3ca-4828-840c-110585e54a9f/csds', 'csds/2023/12/10/b186af28-08fa-41b2-bced-6445f8d2c39b/csds', 'csds/2023/12/10/bce3c6d3-93a0-4f59-8b6c-0cc441392e19/csds', 'csds/2023/12/10/c628aa80-20ec-456f-90e2-02e348b1191c/csds', 'csds/2023/12/10/c856c449-707a-43ff-8d9c-3ef6cefd4721/csds', 'csds/2023/12/10/d38fcc03-48ae-4184-b1d4-9db479360b76/csds', 'csds/2023/12/10/d81f91db-df63-4ffe-9d72-0c094c2e70df/csds', 'csds/2023/12/10/d8f756b8-c3e5-4f0d-b62d-e50223ebdff2/csds', 'csds/2023/12/10/dc87da83-52a4-4c0d-a86e-7c9350e4ce54/csds', 'csds/2023/12/10/e2e036f6-48db-44cb-aae1-98983633e3d9/csds', 'csds/2023/12/10/e5881b09-b922-43a0-a65b-122b1a409c75/csds', 'csds/2023/12/10/f168d5b1-8ead-493a-b835-33e7519130f4/csds', 'csds/2023/12/10/f3f181c7-fffc-495a-ad94-919ac8014d9e/csds', 'csds/2023/12/10/f5387172-ab69-4887-a31c-c4bd651ddc2a/csds', 'csds/2023/12/10/f90296f6-9ace-4a48-a9d8-baf5a2a02b1b/csds', 'csds/2023/12/11/0706d00b-ff59-4a78-900f-4c2ffecf2d14/csds', 'csds/2023/12/11/09d93ef4-ade2-431b-a550-25c0f42c8c16/csds', 'csds/2023/12/11/0c7cfaec-0e3e-4852-908e-df2b7dcac2f3/csds', 'csds/2023/12/11/0d917bf7-2c0b-44f2-b584-3babbf35c2c6/csds', 'csds/2023/12/11/1d818114-9fa1-4656-97ec-fb2f64acb1ea/csds', 'csds/2023/12/11/2078f45a-b80d-4685-b5e6-be8398cd7249/csds', 'csds/2023/12/11/2588f507-1cb4-41fd-932d-8044e52b2014/csds', 'csds/2023/12/11/364ad321-19a9-494b-887b-dd6027b27d63/csds', 'csds/2023/12/11/436da7b2-5527-4db7-9462-eb305775aa1b/csds', 'csds/2023/12/11/47d98bca-e4c5-4c0e-a4eb-00d65f312a60/csds', 'csds/2023/12/11/4e399754-977b-48ca-834c-dddd65340e2b/csds', 'csds/2023/12/11/6f0f4416-4485-4d6b-9a80-79986784d73c/csds', 'csds/2023/12/11/756d0fdd-3a37-4279-8761-e9ceb0c3e647/csds', 'csds/2023/12/11/7968ac4e-c2ff-4e19-a963-62c726714c51/csds', 'csds/2023/12/11/7d51c8ad-ff4f-44ca-860d-b74d8a143fa8/csds', 'csds/2023/12/11/800150ee-732f-4484-ba40-5f647eb3d1e5/csds', 'csds/2023/12/11/825c4d01-12fa-482b-aea9-c6f05e5a0eae/csds', 'csds/2023/12/11/82f6a847-ccc6-4c85-bda4-7e37fa794cfa/csds', 'csds/2023/12/11/915d010d-b7c3-4279-bbec-22971746ebc8/csds', 'csds/2023/12/11/9c8ceaf7-0c7f-4c1d-9b0c-9ef9207ff165/csds', 'csds/2023/12/11/a00c92e9-8d43-4fc2-b5f9-81a55e6e84ec/csds', 'csds/2023/12/11/a76f5b74-c1c0-484e-82e3-4c2737c37c08/csds', 'csds/2023/12/11/a857f245-5a48-4d6b-af35-f99e2be0b10a/csds', 'csds/2023/12/11/ae6d0b9a-0b59-4847-afe8-ed7a391c79b0/csds', 'csds/2023/12/11/b2b48d50-91e0-414a-8e32-9232959e3c00/csds', 'csds/2023/12/11/b582e354-5eb0-4608-bcd2-3751809f4e25/csds', 'csds/2023/12/11/bb03f650-6639-466f-af7e-f88fc1cb6b2d/csds', 'csds/2023/12/11/c216c5e8-f554-4e91-bd67-d1343579a7dc/csds', 'csds/2023/12/11/c921b59f-d0c4-4890-ac4c-258bb6cdcbfb/csds', 'csds/2023/12/11/db5c4ba8-22f2-4de3-97df-5f938f078828/csds', 'csds/2023/12/11/e3b504ae-7057-46ab-ab19-a8cc9f6a1193/csds', 'csds/2023/12/11/e50498b0-219e-4dd9-8cb7-65e04e003bb7/csds', 'csds/2023/12/11/f0f6c2e1-ad85-4bf7-b04a-bb71006cbaef/csds', 'csds/2023/12/11/f1462abb-62c4-4cd2-9bd9-dc8854b60f96/csds', 'csds/2023/12/11/f1e7e634-2afe-47f7-aae0-7fd2b3722149/csds', 'csds/2023/12/11/f2ca02f8-8d31-47e7-aef6-7072ce6efdb0/csds', 'csds/2023/12/12/06253c00-c4a5-4f4f-92e0-129116504b23/csds', 'csds/2023/12/12/062f71aa-5361-4516-b22d-4ef221cf96cb/csds', 'csds/2023/12/12/06c6a854-9a6f-4e1b-ade3-0f31cbdc8970/csds', 'csds/2023/12/12/362551ad-a576-404c-a23f-56b6ff29aca2/csds', 'csds/2023/12/12/5e59255e-b752-4ed7-a4cf-550619283a2b/csds', 'csds/2023/12/12/69c13bc0-0458-4755-84e4-f8e57088636f/csds', 'csds/2023/12/12/7db002c1-28fe-43d5-986e-9cbc23397b79/csds', 'csds/2023/12/12/8051b3a3-7ea2-4d3b-88f1-cc0a35636dd8/csds', 'csds/2023/12/12/829f3105-4424-4079-bbcb-acfe7c3c36fb/csds', 'csds/2023/12/12/89bc37a1-83b8-47a6-836e-b16938012bb2/csds', 'csds/2023/12/12/8ec9ecb8-7a93-43b3-90ee-ceeec82d9cff/csds', 'csds/2023/12/12/8ef9fcbd-d7d1-427d-8fcf-7dc9e264e503/csds', 'csds/2023/12/12/c41d54dd-4109-441d-9f35-6e3af7dc20e4/csds', 'csds/2023/12/12/ca3fcb12-0c40-42c7-bb7d-44c3d02e44fb/csds', 'csds/2023/12/12/def002c5-6626-4eef-af88-831e3265488f/csds', 'csds/2023/12/13/01a18da7-e791-4da8-870b-f86b86035398/csds', 'csds/2023/12/13/24ab0d22-b783-448e-8e10-fc6d3f241eb2/csds', 'csds/2023/12/13/25446c45-ad7d-4339-8fa8-11ee249f5ebb/csds', 'csds/2023/12/13/285adbd2-ef84-4caa-b358-a100f9d05767/csds', 'csds/2023/12/13/32a5ad7d-990c-4416-99a7-ffbbb9ac7363/csds', 'csds/2023/12/13/34e18b39-9720-421f-961c-4dd253f112ec/csds', 'csds/2023/12/13/37580530-b5f7-4043-8b9e-0c83df27c6f4/csds', 'csds/2023/12/13/5d179cf5-f51c-4762-8e8c-bf43a6ef8989/csds', 'csds/2023/12/13/76aba571-c77c-4e10-9606-1c05062d856f/csds', 'csds/2023/12/13/7d461d04-d772-4acb-8137-e2893686b35c/csds', 'csds/2023/12/13/c18ca679-7cb9-4cfe-8d9d-22ec43c4a8c5/csds', 'csds/2023/12/13/c23b9b19-c369-49cc-a3eb-8f437eace051/csds', 'csds/2023/12/13/cbcc63cc-7f08-4074-8216-8d4838ac5ffd/csds']


def train_global_kmeans(match_generator, target_seconds=28, time_window=3, tick_rate=64):
    all_positions = []

    for match_data in match_generator:
        df_vector = match_data['player_vector'].copy()
        df_round_start = match_data['round_start']

        df_vector['round'] = df_vector['round'] - 1 
        round_starts = df_round_start[['round', 'tick']].rename(columns={'tick': 'start_tick'})
        
        df_merged = pd.merge(df_vector, round_starts, on='round', how='inner')
        df_ct_early = df_merged[df_merged['team_code'] == 3].copy()

        window_start = df_ct_early['start_tick'] + ((target_seconds - time_window) * tick_rate)
        window_end = df_ct_early['start_tick'] + (target_seconds * tick_rate)
        df_window = df_ct_early[(df_ct_early['tick'] >= window_start) & (df_ct_early['tick'] <= window_end)]
        
        # 只保留這場比賽的 x, y, z 平均座標
        df_ct_positions = df_window.groupby(['round', 'player_id'])[['x_pos', 'y_pos', 'z_pos']].mean().reset_index()
        all_positions.append(df_ct_positions[['x_pos', 'y_pos', 'z_pos']])

    # 將多場比賽的座標合併成一個 DataFrame
    df_all_positions = pd.concat(all_positions, ignore_index=True)
    print(f"K-Means 訓練完成，共使用 {len(df_all_positions)} 筆玩家開局站位座標。")

    kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
    kmeans.fit(df_all_positions[['x_pos', 'y_pos', 'z_pos']])
    return kmeans



def main():
    all_transactions = []

    train_generator = data_loader.get_match_with_key_mask(KEY_MASK_DUST2, max_matches=10)
    global_kmeans = train_global_kmeans(train_generator)

    # first_match_data = next(data_loader.get_match_with_key_mask(KEY_MASK_DUST2, max_matches=1))
    # global_kmeans = train_global_kmeans(first_match_data)
    random_designated_match = random.randint(2,60)
    cnt = 0
    for match_data in data_loader.get_match_with_key_mask(KEY_MASK_DUST2, max_matches=100):
    # match_list = data_loader.get_match_with_key_mask(KEY_MASK_DUST2)
        cnt +=1
        if cnt == random_designated_match:
            match_transactions = opening_positioning_rule.start_analyze(match_data,global_kmeans,True, verbose=False)
        else:
            match_transactions = opening_positioning_rule.start_analyze(match_data,global_kmeans,False, verbose=False)
        all_transactions.extend(match_transactions)
    print(f"\n Match data collected, Number of round recorded: {len(all_transactions)}")

    # differnet set of minsup and minconf
    # minsup = 0.15, minconf = 0.06 -> a lot of same tactics (zone_2 m4a1, zone_2 hegrenade)
    if len(all_transactions):
        opening_positioning_rule.run_apriori(all_transactions, min_support=0.1, min_confidence=0.5)
    


def train_global_kmeans_v1(match_data, target_seconds=28, time_window=3, tick_rate=64):
    df_vector = match_data['player_vector']
    df_round_start = match_data['round_start']

    df_vector['round'] = df_vector['round'] - 1 

    round_starts = df_round_start[['round', 'tick']].rename(columns={'tick': 'start_tick'})
    
    # calculate target_tick
    round_starts['target_tick'] = round_starts['start_tick'] + (target_seconds * tick_rate)

    
    df_merged = pd.merge(df_vector, round_starts, on='round', how='inner')
    
    df_ct_early = df_merged[df_merged['team_code'] == 3].copy() # 3 = CT
    # # there's a 15 second idle time, shop time??

    # get average position on (target second - time_window)
    window_start = df_ct_early['start_tick'] + ((target_seconds - time_window) * tick_rate)
    window_end = df_ct_early['start_tick'] + (target_seconds * tick_rate)
    df_window = df_ct_early[(df_ct_early['tick'] >= window_start) & (df_ct_early['tick'] <= window_end)]
    df_ct_positions = df_window.groupby(['round', 'player_id'])[['x_pos', 'y_pos', 'z_pos']].mean().reset_index()


    # 4. 利用 K-Means 將座標分為 "n_clusters" 個防守區域，給予每個座標一個區域標籤 (Cluster_0 ~ Cluster_n)
    # K=7: A platform, A tunnel, A arc, CT spawn/CT door, B bomb, B tunnel, A tunnel close to T side
    kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
    kmeans.fit(df_ct_positions[['x_pos', 'y_pos', 'z_pos']])
    return kmeans

if __name__ == "__main__":
    main()




