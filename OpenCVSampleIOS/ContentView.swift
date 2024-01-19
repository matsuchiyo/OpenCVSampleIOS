//
//  ContentView.swift
//  OpenCVSampleIOS
//
//  Created by 松島勇貴 on 2023/10/12.
//

import SwiftUI
import opencv2

struct ContentView: View {
    var body: some View {
        /*
        VStack {
            Image(uiImage: RectangleDetector2.detectRectangle(image:  UIImage(named: "receipt")!) ?? UIImage.actions)
                .resizable()
                .scaledToFit()
                .frame(width: 360)
            Text("Hello, world!")
        }
        .padding()
         */
        List(ReceiptDetector.detect(image: UIImage(named: "receipt_with_white_background")!, returnProcessingImages: true).processingImages, id: \.self) { item in
            Image(uiImage: item)
                .resizable()
                .scaledToFit()
                .frame(width: 360)
        }
    }
    
    func convertColor(source srcImage: UIImage) -> UIImage {
        let srcMat = Mat(uiImage: srcImage)
        let dstMat = Mat()
        Imgproc.cvtColor(src: srcMat, dst: dstMat, code: .COLOR_RGB2GRAY)
        return dstMat.toUIImage()
    }}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
