//
//  RectangleDetecter.swift
//  OpenCVSampleIOS
//
//  Created by 松島勇貴 on 2023/10/13.
//

import UIKit
import opencv2

class RectangleDetector {
    static func detectRectangle(image: UIImage) -> UIImage? {
        let matSource = Mat(uiImage: image)
        print("***** matSource \(matSource.rows()) \(matSource.cols()) \(matSource.type()) ")
        let matDest = Mat()
        // グレースケール変換
        Imgproc.cvtColor(src: matSource, dst: matDest, code: .COLOR_BGR2GRAY)
        // return matDest.toUIImage()
        // 2値化
        
        /*
        // https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html の Otsu's Binarizationを参考に、事前にガウシアンブラーをかけてみる。
//        Imgproc.GaussianBlur(src: matDest, dst: matDest, ksize: Size2i(width: 5, height: 5), sigmaX: 5, sigmaY: 5)
        
        let thresholdTypes: ThresholdTypes = ThresholdTypes(rawValue: ThresholdTypes.THRESH_BINARY.rawValue | ThresholdTypes.THRESH_OTSU.rawValue)!
//        let thresholdTypes = ThresholdTypes.THRESH_BINARY
        Imgproc.threshold(src: matDest, dst: matDest, thresh: 0, maxval: 255, type: thresholdTypes)
//        Imgproc.threshold(src: matDest, dst: matDest, thresh: 16, maxval: 255, type: thresholdTypes)
        
         */
        ///*
        // adaptive thresholdを使ってみる
        Imgproc.adaptiveThreshold(src: matDest, dst: matDest, maxValue: 255, adaptiveMethod: AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType: ThresholdTypes.THRESH_BINARY, blockSize: 11, C: 2)
        // */
//        return matDest.toUIImage() // 検証用
        
        // /*
        // https://www.kaggle.com/code/dmitryyemelyanov/receipt-ocr-part-1-image-segmentation-by-opencv
//        Imgproc.GaussianBlur(src: matDest, dst: matDest, ksize: Size2i(width: 5, height: 5), sigmaX: 0)
        Imgproc.GaussianBlur(src: matDest, dst: matDest, ksize: Size2i(width: 51, height: 51), sigmaX: 25)
//        return matDest.toUIImage()
        let kernelForDilation = Imgproc.getStructuringElement(shape: MorphShapes.MORPH_RECT, ksize: Size2i(width: 9, height: 9))
        Imgproc.dilate(src: matDest, dst: matDest, kernel: kernelForDilation)
        
        
        // */
        
        
        let thresholdTypes = ThresholdTypes.THRESH_BINARY
        Imgproc.threshold(src: matDest, dst: matDest, thresh: 192, maxval: 255, type: thresholdTypes)
//        return matDest.toUIImage()
        
        
        // Cannyアルゴリズムを使ったエッジ検出
        let matCanny = Mat()
        Imgproc.Canny(image: matDest, edges: matCanny, threshold1: 75, threshold2: 200)
//        Imgproc.Canny(image: matDest, edges: matCanny, threshold1: 100, threshold2: 200, apertureSize: 3)
//        return matCanny.toUIImage() // 検証用
        // 膨張
        let matKernel = Imgproc.getStructuringElement(shape: MorphShapes.MORPH_RECT, ksize: Size2i(width: 9, height: 9))
        Imgproc.dilate(src: matCanny, dst: matCanny, kernel: matKernel)
//        return matCanny.toUIImage() // 検証用
        
        
        // 輪郭を取得
        let vctContours: NSMutableArray = [] // Point2iのNSMutableArray(=contour(輪郭))のNSMutableArray
        let hierarychy = Mat()
        Imgproc.findContours(image: matCanny, contours: vctContours, hierarchy: hierarychy, mode: RetrievalModes.RETR_LIST, method: ContourApproximationModes.CHAIN_APPROX_SIMPLE)
        
        // 面積順にソート(大きい順)
        vctContours.sort(comparator: { contour1, contour2 in
            // TODO: 不安
//            let contour1Mat: Mat = MatOfPoint(array: contour1 as! [Point2i])
            let contour1Mat: Mat = intPointsToMat(points: contour1 as! [Point2i])
//            let contour2Mat: Mat = MatOfPoint(array: contour2 as! [Point2i])
            let contour2Mat: Mat = intPointsToMat(points: contour2 as! [Point2i])
            return Imgproc.contourArea(contour: contour1Mat, oriented: false) > Imgproc.contourArea(contour: contour2Mat, oriented: false) ? ComparisonResult.orderedAscending : ComparisonResult.orderedDescending
        })
        
        // TODO: delete
        vctContours.forEach {
//            let contour1Mat: Mat = MatOfPoint(array: $0 as! [Point2i])
            let contour1Mat: Mat = intPointsToMat(points: $0 as! [Point2i])
            print("***** Imgproc.contourArea(contour: contour1Mat, oriented: false): \(Imgproc.contourArea(contour: contour1Mat, oriented: false))")
        }
        
        // 検証のため、検出した部分を四角で囲む。
        let matCanny3Channel = Mat()
        Imgproc.cvtColor(src: matCanny, dst: matCanny3Channel, code: .COLOR_GRAY2RGB)
        for i in 0..<min(10, vctContours.count) {
            let contour = vctContours[i]
            let minX: Int32 = (contour as! [Point2i]).min(by: { $0.x > $1.x })!.x
            let maxX: Int32 = (contour as! [Point2i]).max(by: { $0.x > $1.x })!.x
            let minY: Int32 = (contour as! [Point2i]).min(by: { $0.y > $1.y })!.y
            let maxY: Int32 = (contour as! [Point2i]).max(by: { $0.y > $1.y })!.y
            Imgproc.rectangle(img: matCanny3Channel, pt1: Point2i(x: minX, y: minY), pt2: Point2i(x: maxX, y: maxY),  color: Scalar(0, 255, 0), thickness: 24)
        }
//        return matCanny3Channel.toUIImage()
        
        // 最大の四角形を走査、変換元の矩形にする
        let ptSrc: NSMutableArray = []
        for i in 0..<vctContours.count {
            let contour = vctContours[i]
            let approxCurve: NSMutableArray = []
            let curve = (contour as! [Point2i]).map { Point2f(x: Float($0.x), y: Float($0.y)) } as! [Point2f]
            let arclen = Imgproc.arcLength(curve: curve, closed: true)
            Imgproc.approxPolyDP(curve: curve, approxCurve: approxCurve, epsilon: 0.025 * arclen, closed: true)
            // 4辺の矩形なら採用
            if (approxCurve.count == 4) {
                print("***** first rectangle index: \(i)")
                for j in 0..<4 {
                    let approxCurvePoint = approxCurve[j]
                    let point = approxCurvePoint as! Point2f
                    ptSrc.add(Point2f(x: point.x, y: point.y))
                }
                break // 一番大きい矩形だけ使う
            }
        }
        
        if (ptSrc.count == 0) {
            return nil
        }
        
        // 変換先の矩形（元画像の幅を最大にした名刺比率にする）
        let width: Float = Float(image.size.width);
        let height: Float = width / 1.654 // TODO:
        let ptDst: NSMutableArray = [
            Point2f(x: 0, y: height),
            Point2f(x: width, y: height),
            Point2f(x: width, y: 0),
            Point2f(x: 0, y: 0),
        ]
        
        // 変換行列
//        let ptSrcMat = MatOfPoint(array: (ptSrc as! [Point2f]).map { Point2i(x: Int32($0.x), y: Int32($0.y)) })
//        let ptDstMat = MatOfPoint(array: (ptDst as! [Point2f]).map { Point2i(x: Int32($0.x), y: Int32($0.y)) })
//        let ptSrcMat = intPointsToMat(points: (ptSrc as! [Point2f]).map { Point2i(x: Int32($0.x), y: Int32($0.y)) })
//        let ptDstMat = intPointsToMat(points: (ptDst as! [Point2f]).map { Point2i(x: Int32($0.x), y: Int32($0.y)) })
        let ptSrcMat = floatPointsToMat(points: (ptSrc as! [Point2f]))
        let ptDstMat = floatPointsToMat(points: (ptDst as! [Point2f]))
        print("ptSrcMat rows: \(ptSrcMat.rows()) cols: \(ptSrcMat.cols())")
        print("ptDstMat rows: \(ptDstMat.rows()) cols: \(ptDstMat.cols())")
        let matTrans = Imgproc.getPerspectiveTransform(src: ptSrcMat, dst: ptDstMat)
        
        // 変換
        // TODO: 不安。rowsとcols逆じゃない？
//        let matResult = Mat(rows: Int32(width), cols: Int32(height), type: matSource.type())
        let matResult = Mat(rows: Int32(height), cols: Int32(width), type: matSource.type())
        Imgproc.warpPerspective(src: matSource, dst: matResult, M: matTrans, dsize: Size2i(width: matResult.width(), height: matResult.height()))
        
        return matResult.toUIImage()
    }
    
    
    private static func intPointsToMat(points: [Point2i]) -> Mat {
        let CV_32S_C1 = Int32(4) // 32ビット符号付き整数、1チャネル
        let mat = Mat(rows: Int32(points.count), cols: 2, type: CV_32S_C1)
        for i in 0..<points.count {
            mat.at(row: Int32(i), col: Int32(0)).v = points[i].x
            mat.at(row: Int32(i), col: Int32(1)).v = points[i].y
        }
        return mat
    }
    
    
    private static func floatPointsToMat(points: [Point2f]) -> Mat {
        let CV_32F_C1 = Int32(5) // 32ビット符号付き浮動小数点、1チャネル
        let mat = Mat(rows: Int32(points.count), cols: 2, type: CV_32F_C1)
        for i in 0..<points.count {
            mat.at(row: Int32(i), col: Int32(0)).v = points[i].x
            mat.at(row: Int32(i), col: Int32(1)).v = points[i].y
        }
        return mat
    }
}
