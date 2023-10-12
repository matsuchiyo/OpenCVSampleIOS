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
        VStack {
            Image(uiImage: UIImage(named: "image")!)
                .resizable()
                .frame(width: 200, height: 200)
            Image(uiImage: convertColor(source: UIImage(named: "image")!))
                .resizable()
                .frame(width: 200, height: 200)
            Text("Hello, world!")
        }
        .padding()
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
