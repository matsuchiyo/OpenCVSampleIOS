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
        
        let width = Int32(image.size.width)
        let height = Int32(image.size.height)
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
        if returnProcessingImages { processingImages.append(contentsOf: resultByReceiptEdges.processingImages) }
        print("***** contour1 exists: \(resultByReceiptEdges.contour != nil)")
        if let contour = resultByReceiptEdges.contour {
            let contourArea = OpenCVUtils.contourArea(contour)
            print("***** contourArea1: \(contourArea)")
            if contourArea > contourAreaThreshold {
                let extractedImage = OpenCVUtils.extractImageByContour(image: image, receiptContour: contour)
                if returnProcessingImages { processingImages.append(extractedImage) }
                return ReceiptDetectResult(result: extractedImage, processingImages: processingImages)
            }
        }
        
        let resultByReceiptContent = ReceiptContourDetectorByContent.detect(image: image, returnProcessingImages: returnProcessingImages)
        if returnProcessingImages { processingImages.append(contentsOf: resultByReceiptContent.processingImages) }
        print("***** contour2 exists: \(resultByReceiptContent.contour != nil)")
        if let contour = resultByReceiptContent.contour {
            let contourArea = OpenCVUtils.contourArea(contour)
            print("***** contourArea2: \(contourArea)")
            if contourArea > contourAreaThreshold {
                let extractedImage = OpenCVUtils.extractImageByContour(image: image, receiptContour: contour)
                if returnProcessingImages { processingImages.append(extractedImage) }
                return ReceiptDetectResult(result: extractedImage, processingImages: processingImages)
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
                var convexHullAppliedLargestContour: [Point2i] = []
                for i in 0..<indicesOfPointsWhichComposeHull.length {
                    convexHullAppliedLargestContour.append($0[Int(indicesOfPointsWhichComposeHull[i])])
                }
                return convexHullAppliedLargestContour
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
        
        Imgproc.GaussianBlur(src: imageMat, dst: imageMat, ksize: Size2i(width: 31, height: 31), sigmaX: 0) // 可能な限りノイズを除去。かつ文字を残す(存在するのがわかる程度)。
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

class OpenCVUtils {
    static func resize(mat: Mat, resizeRatio: Double) {
        let width = Int32(Double(mat.width()) * resizeRatio)
        let height = Int32(Double(mat.height()) * resizeRatio)
        Imgproc.resize(src: mat, dst: mat, dsize: Size2i(width: width, height: height))
    }
    
    
    static func contourArea(_ contour: [Point2i]) -> Double {
        let contourMat = intPointsToMat(points: contour)
        return Imgproc.contourArea(contour: contourMat, oriented: false)
    }
    
    // contourArea()が、配列ではなくMatで受けるようになっている。そのため、Point(x, y)の配列を、2列・[Point数]行 の行列に変換。
    private static func intPointsToMat(points: [Point2i]) -> Mat {
        let CV_32S_C1 = Int32(4) // 32ビット符号付き整数、1チャネル
        let mat = Mat(rows: Int32(points.count), cols: 2, type: CV_32S_C1)
        for i in 0..<points.count {
            mat.at(row: Int32(i), col: Int32(0)).v = points[i].x
            mat.at(row: Int32(i), col: Int32(1)).v = points[i].y
        }
        return mat
    }
    
    // contourArea()が、配列ではなくMatで受けるようになっている。そのため、Point(x, y)の配列を、2列・[Point数]行 の行列に変換。
    private static func intPointsToFloat32Mat(points: [Point2i]) -> Mat {
        let CV_32F_C1 = Int32(5) // 32ビット符号付き整数、1チャネル
        let mat = Mat(rows: Int32(points.count), cols: 2, type: CV_32F_C1)
        for i in 0..<points.count {
            mat.at(row: Int32(i), col: Int32(0)).v = Float(points[i].x)
            mat.at(row: Int32(i), col: Int32(1)).v = Float(points[i].y)
        }
        return mat
    }
    
    
    static func getFirst4CornerContour(contours: [[Point2i]]) -> [Point2i]? {
        return contours.first { contour in
            let approx = approximateContour(contour: contour)
            print("***** approx.counnt: \(approx.count)")
            return approx.count == 4
        }
    }
    
    private static func approximateContour(contour: [Point2i]) -> [Point2f] {
        let contourInPoint2f = point2isToPoint2fs(contour)
        let perimeterLength = Imgproc.arcLength(curve: contourInPoint2f, closed: true)
        let approxContour = NSMutableArray()
        Imgproc.approxPolyDP(curve: contourInPoint2f, approxCurve: approxContour, epsilon: 0.032 * perimeterLength, closed: true)
        print("***** approximateContour: \(approxContour)")
        return approxContour as! [Point2f]
    }
    
    private static func point2isToPoint2fs(_ point2is: [Point2i]) -> [Point2f] {
        return point2is.map { point2iToPoint2f($0) }
    }
    
    private static func point2iToPoint2f(_ point2i: Point2i) -> Point2f {
        return Point2f(
            x: Float(point2i.x),
            y: Float(point2i.y)
        )
    }
    
    static func extractImageByContour(image: UIImage, receiptContour: [Point2i]) -> UIImage {
        let original = Mat(uiImage: image)
        let receiptRect = contourToRect(contours: receiptContour)
        let transformedMat = warpPerspective(imageMat: original, rect: receiptRect)

        let grayScaledMat = Mat()
        Imgproc.cvtColor(src: transformedMat, dst: grayScaledMat, code: .COLOR_BGR2GRAY)
        Imgproc.adaptiveThreshold(src: grayScaledMat, dst: grayScaledMat, maxValue: 255, adaptiveMethod: AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType: ThresholdTypes.THRESH_BINARY, blockSize: 21, C: 5)

        return grayScaledMat.toUIImage()
    }
    
    private static func contourToRect(contours: [Point2i]) -> [Point2i] {
        let xySumPerPointList: [Int32] = contours.map { $0.x + $0.y }
        let maxXySumPerPointList = xySumPerPointList.max()!
        let maxIndexXySumPerPointList: Int = xySumPerPointList.firstIndex(of: maxXySumPerPointList)!
        let minXySumPerPointList = xySumPerPointList.min()!
        let minIndexXySumPerPointList: Int = xySumPerPointList.firstIndex(of: minXySumPerPointList)!
        
        let yxDiffPerPointList: [Int32] = contours.map { $0.y - $0.x }
        let maxYxDiffPerPointList = yxDiffPerPointList.max()!
        let maxIndexYxDiffPerPointList = yxDiffPerPointList.firstIndex(of: maxYxDiffPerPointList)!
        let minYxDiffPerPointList = yxDiffPerPointList.min()!
        let minIndexYxDiffPerPointList = yxDiffPerPointList.firstIndex(of: minYxDiffPerPointList)!
        
        return [
            contours[minIndexXySumPerPointList],
            contours[minIndexYxDiffPerPointList],
            contours[maxIndexXySumPerPointList],
            contours[maxIndexYxDiffPerPointList],
        ]
    }
    
    private static func warpPerspective(imageMat: Mat, rect: [Point2i]) -> Mat {
        let topLeft = rect[0]
        let topRight = rect[1]
        let bottomRight = rect[2]
        let bottomLeft = rect[3]
        
        let widthA = distance(bottomRight, bottomLeft)
        let widthB = distance(topRight, topLeft)
        let heightA = distance(topRight, bottomRight)
        let heightB = distance(topLeft, bottomLeft)
        
        let maxWidth = max(widthA, widthB)
        let maxHeight = max(heightA, heightB)
        
        let transformMat = Imgproc.getPerspectiveTransform(
            src: intPointsToFloat32Mat(points: rect),
            dst: intPointsToFloat32Mat(points: [
                Point2i(x: 0, y: 0),
                Point2i(x: Int32(maxWidth - 1), y: 0),
                Point2i(x: Int32(maxWidth - 1), y: Int32(maxHeight - 1)),
                Point2i(x: 0, y: Int32(maxHeight - 1)),
            ])
        )
        
        let resultMat = Mat()
        Imgproc.warpPerspective(
            src: imageMat,
            dst: resultMat,
            M: transformMat,
            dsize: Size2i(
                width: Int32(maxWidth),
                height: Int32(maxHeight)
            )
        )
        return resultMat
    }
    
    private static func distance(_ p1: Point2i, _ p2: Point2i) -> Int {
        return Int(sqrt(pow(Double(p1.x - p2.x), 2.0)) + sqrt(pow(Double(p1.y - p2.y), 2.0)))
    }
    
    
    // debug用
    static func imageWithContours(image: Mat, contours: [[Point2i]], contourScale: Double = 1.0) -> UIImage {
        let imageWithContours = Mat()
        image.copy(to: imageWithContours)
        let resizedContours = contours.map { contour in
            return contour.map { point in
                return Point2i(
                    x: Int32(Double(point.x) * contourScale),
                    y: Int32(Double(point.y) * contourScale)
                )
            }
        }
        
        for resizedContour in resizedContours {
            Imgproc.drawContours(
                image: imageWithContours,
                contours: [resizedContour],
                contourIdx: -1,
    //            color: Scalar(0, 255, 0, 255),
                color: Scalar(
                    Double.random(in: 0...255),
                    Double.random(in: 0...255),
                    Double.random(in: 0...255),
                    255
                ),
    //            thickness: 24
                thickness: 12
            )
        }
        return imageWithContours.toUIImage()
    }
}
