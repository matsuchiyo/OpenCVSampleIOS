//
//  ReceiptDetector.swift
//  OpenCVSampleIOS
//
//  Created by 松島勇貴 on 2024/01/19.
//

import Foundation
import opencv2

struct ReceiptDetectResult {
    let result: UIImage?
    let processingImages: [UIImage]
}

struct ContourDetectResult {
    let contour: [Point2i]?
    let processingImages: [UIImage]
}

class ReceiptDetector {
    static func detect(image: UIImage, returnProcessingImages: Bool) -> ReceiptDetectResult {
        var processingImages: [UIImage] = []
        if returnProcessingImages { processingImages.append(image) }
        
        let imageMat = Mat(uiImage: image)
        
        let width = imageMat.width()
        let height = imageMat.height()
        let contourThatFillsImage = [
            Point2i(x: 0, y: 0),
            Point2i(x: width - 1, y: 0),
            Point2i(x: width - 1, y: height - 1),
            Point2i(x: 0, y: height - 1),
            Point2i(x: 0, y: 0),
        ]
        let areaOfContourThatFillsImage = OpenCVUtils.contourArea(contourThatFillsImage)
        print("***** areaOfContourThatFillsImage: \(areaOfContourThatFillsImage)")
        let contourAreaThreshold = areaOfContourThatFillsImage / 10
        
        let resultByReceiptEdges = ReceiptContourDetectorByReceiptEdges.detect(image: image, convexHull: false, returnProcessingImages: returnProcessingImages)
        processingImages.append(contentsOf: resultByReceiptEdges.processingImages)
        print("***** contour1 exists: \(resultByReceiptEdges.contour != nil)")
        if let contour = resultByReceiptEdges.contour {
            let contourArea = OpenCVUtils.contourArea(contour)
            print("***** contourArea1: \(contourArea)")
            if contourArea > contourAreaThreshold {
                let image = OpenCVUtils.extractImageByContour(image: image, receiptContour: contour)
                if returnProcessingImages { processingImages.append(image) }
                return ReceiptDetectResult(result: image, processingImages: processingImages)
            }
        }
        
        let resultByReceiptContent = ReceiptContourDetectorByContent.detect(image: image, returnProcessingImages: returnProcessingImages)
        processingImages.append(contentsOf: resultByReceiptContent.processingImages)
        print("***** contour2 exists: \(resultByReceiptContent.contour != nil)")
        if let contour = resultByReceiptContent.contour {
            let contourArea = OpenCVUtils.contourArea(contour)
            print("***** contourArea2: \(contourArea)")
            if contourArea > contourAreaThreshold {
                let image = OpenCVUtils.extractImageByContour(image: image, receiptContour: contour)
                if returnProcessingImages { processingImages.append(image) }
                return ReceiptDetectResult(result: image, processingImages: processingImages)
            }
        }
        
        return ReceiptDetectResult(result: nil, processingImages: processingImages)
    }
}

class ReceiptContourDetectorByReceiptEdges {
    static func detect(image: UIImage, convexHull: Bool, returnProcessingImages: Bool) -> ContourDetectResult {
        var processingImages: [UIImage] = []
        
        let original = Mat(uiImage: image)
        let imageMat = Mat()
        original.copy(to: imageMat)
        
        let resizeRatio = 500.0 / Double(imageMat.height())
        OpenCVUtils.resize(mat: imageMat, resizeRatio: resizeRatio)
        
        Imgproc.cvtColor(src: imageMat, dst: imageMat, code: .COLOR_BGR2GRAY)
        
        Imgproc.GaussianBlur(src: imageMat, dst: imageMat, ksize: Size2i(width: 5, height: 5), sigmaX: 0) // ノイズの除去
        if returnProcessingImages { processingImages.append(imageMat.toUIImage()) }
        
        let kernelForDilation = Imgproc.getStructuringElement(shape: MorphShapes.MORPH_RECT, ksize: Size2i(width: 9, height: 9))
        Imgproc.dilate(src: imageMat, dst: imageMat, kernel: kernelForDilation)
        if returnProcessingImages { processingImages.append(imageMat.toUIImage()) }
        
        let edgesMat = Mat()
        Imgproc.Canny(image: imageMat, edges: edgesMat, threshold1: 100, threshold2: 200, apertureSize: 3)
        if returnProcessingImages { processingImages.append(edgesMat.toUIImage()) }
        
        let contours: NSMutableArray = []
        let hierarchy = Mat()
        Imgproc.findContours(image: edgesMat, contours: contours, hierarchy: hierarchy, mode: .RETR_TREE, method: .CHAIN_APPROX_SIMPLE)
        
        let contourArray = contours.copy() as! [[Point2i]]
        print("***** contourArray.count: \(contourArray.count)")
        
        let sortedContourArray = contourArray.sorted(by: { OpenCVUtils.contourArea($0) > OpenCVUtils.contourArea($1) })
        var largestContours: [[Point2i]] = Array(sortedContourArray.prefix(10))
        if returnProcessingImages { processingImages.append(OpenCVUtils.imageWithContours(image: original, contours: largestContours, contourScale: 1.0 / resizeRatio)) }
        
        if convexHull {
            largestContours = largestContours.map {
                let indicesOfPointsWhichComposeHull = IntVector()
                Imgproc.convexHull(points: $0, hull: indicesOfPointsWhichComposeHull)
                var convexHullAppliedLargestContours: [Point2i] = []
                for i in 0..<indicesOfPointsWhichComposeHull.length {
                    convexHullAppliedLargestContours.append($0[Int(indicesOfPointsWhichComposeHull[i])])
                }
                return convexHullAppliedLargestContours
            }
            
            if returnProcessingImages { processingImages.append(OpenCVUtils.imageWithContours(image: original, contours: largestContours, contourScale: 1.0 / resizeRatio)) }
        }
        
        let receiptContour: [Point2i]? = OpenCVUtils.getFirst4CornerContour(contours: largestContours)
        let sizeRestoredReceiptContour = receiptContour?.map { Point2i(
            x: Int32(Double($0.x) * (1.0 / resizeRatio)),
            y: Int32(Double($0.y) * (1.0 / resizeRatio))
        ) }
        return ContourDetectResult(contour: sizeRestoredReceiptContour, processingImages: processingImages)
    }
}

class ReceiptContourDetectorByContent {
    static func detect(image: UIImage, returnProcessingImages: Bool) -> ContourDetectResult {
        var processingImages: [UIImage] = []
        
        let original = Mat(uiImage: image)
        let imageMat = Mat()
        original.copy(to: imageMat)
        
        let resizeRatio = 500.0 / Double(imageMat.height())
        OpenCVUtils.resize(mat: imageMat, resizeRatio: resizeRatio)
        
        Imgproc.cvtColor(src: imageMat, dst: imageMat, code: .COLOR_BGR2GRAY)
        
        Imgproc.GaussianBlur(src: imageMat, dst: imageMat, ksize: Size2i(width: 31, height: 31), sigmaX: 0) // ノイズの除去。かつ文字を残す(存在するのがわかる程度)。
        if returnProcessingImages { processingImages.append(imageMat.toUIImage()) }
        
        // 影 削除
        Imgproc.adaptiveThreshold(src: imageMat, dst: imageMat, maxValue: 255, adaptiveMethod: .ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType: .THRESH_BINARY, blockSize: 11, C: 2)
        if returnProcessingImages { processingImages.append(imageMat.toUIImage()) }
        
        let kernelForErosion = Imgproc.getStructuringElement(shape: MorphShapes.MORPH_RECT, ksize: Size2i(width: 61, height: 61))
        Imgproc.erode(src: imageMat, dst: imageMat, kernel: kernelForErosion)
        if returnProcessingImages { processingImages.append(imageMat.toUIImage()) }
        
        Core.bitwise_not(src: imageMat, dst: imageMat) // findContoursのため反転。
        
        let contours: NSMutableArray = []
        let hierarchy = Mat()
        Imgproc.findContours(image: imageMat, contours: contours, hierarchy: hierarchy, mode: .RETR_TREE, method: .CHAIN_APPROX_SIMPLE)
        
        let contourArray = contours.copy() as! [[Point2i]]
        print("***** contourArray.count: \(contourArray.count)")
        
        let sortedContourArray = contourArray.sorted(by: { OpenCVUtils.contourArea($0) > OpenCVUtils.contourArea($1) })
        let largestContours: [[Point2i]] = Array(sortedContourArray.prefix(10))
        
        if returnProcessingImages { processingImages.append(OpenCVUtils.imageWithContours(image: original, contours: largestContours, contourScale: 1.0 / resizeRatio)) }
    
        let convexHullAppliedLargestContours = largestContours.map {
            let indicesOfPointsWhichComposeHull = IntVector()
            Imgproc.convexHull(points: $0, hull: indicesOfPointsWhichComposeHull)
            var convexHullAppliedLargestContours: [Point2i] = []
            for i in 0..<indicesOfPointsWhichComposeHull.length {
                convexHullAppliedLargestContours.append($0[Int(indicesOfPointsWhichComposeHull[i])])
            }
            return convexHullAppliedLargestContours
        }
        
        if returnProcessingImages { processingImages.append(OpenCVUtils.imageWithContours(image: original, contours: convexHullAppliedLargestContours, contourScale: 1.0 / resizeRatio)) }
        
        let receiptContour: [Point2i]? = OpenCVUtils.getFirst4CornerContour(contours: convexHullAppliedLargestContours)
        let sizeRestoredReceiptContour = receiptContour?.map { Point2i(
            x: Int32(Double($0.x) * (1.0 / resizeRatio)),
            y: Int32(Double($0.y) * (1.0 / resizeRatio))
        ) }
        return ContourDetectResult(contour: sizeRestoredReceiptContour, processingImages: processingImages)
    }
}
