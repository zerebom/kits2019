#疑似コード
for patient in patiens:
    for slice in slices:
        #領域
        contours,hierarchy = cv2.findContours(slice, 1, 2)
        
        #最大領域
        max_countor=np.argmax(np.array([cv2.contourArea(cnt) for cnt in contours]))
        rect = cv2.minAreaRect(max_countor)
        
        #回転点,領域の大きさ,回転角
        center,size,degree=rect
        box = cv2.boxPoints(rect)

        horizon_rect=(center,size,0)
        horizon_box = cv2.boxPoints(horizon_rect)
        
        #(Left,Right,Top,Bottom:X,Y)
        LX=horizon_box[0,0]
        RX=horizon_box[3,0]
        TY=horizon_box[0,1]
        UY=horizon_box[1,1]

        #余白のある切り取り領域
        padding_box = np.array([
                     [LX-PADDING,UY+PADDING],
                     [LX-PADDING,TY-PADDING],
                     [RX+PADDING,TY-PADDING],
                     [RX+PADDING,UY+PADDING],
                    ])
        
        #画像全体を回転させる。
        rotate_vol_a=rotate(fix_vol,degree,center)
        rotate_seg_a=rotate(seg,degree,center)
        
        #画像を切り取る
        clip_vol=rotate_vol_a[TY:UY, LX:RX]
        clip_seg=rotate_seg_a[TY:UY, LX:RX]
        
        clip_vol=cv2.resize(clip_vol,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
        clip_seg=cv2.resize(clip_seg,dsize=(256,256),interpolation=cv2.INTER_CUBIC)
        
                padding_box = np.array([
                     [LX-PADDING,UY+PADDING],
                     [LX-PADDING,TY-PADDING],
                     [RX+PADDING,TY-PADDING],
                     [RX+PADDING,UY+PADDING],
                    ])
