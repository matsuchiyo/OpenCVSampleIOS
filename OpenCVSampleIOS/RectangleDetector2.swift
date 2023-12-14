//
//  RectangleDetector2.swift
//  OpenCVSampleIOS
//
//  Created by 松島勇貴 on 2023/12/14.
//

import Foundation
import opencv2

// https://www.kaggle.com/code/dmitryyemelyanov/receipt-ocr-part-1-image-segmentation-by-opencv

class RectangleDetector2 {
    static func detectRectangle(image: UIImage) -> UIImage? {
        let imageMat = Mat(uiImage: image)
        let original = Mat()
        imageMat.copy(to: original)
        
        let resizeRatio = 500.0 / Double(imageMat.height())
        resize(mat: imageMat, resizeRatio: resizeRatio)
        
        Imgproc.cvtColor(src: imageMat, dst: imageMat, code: .COLOR_BGR2GRAY)
        
        Imgproc.GaussianBlur(src: imageMat, dst: imageMat, ksize: Size2i(width: 5, height: 5), sigmaX: 0)
        
        let kernelForDilation = Imgproc.getStructuringElement(shape: MorphShapes.MORPH_RECT, ksize: Size2i(width: 9, height: 9))
        Imgproc.dilate(src: imageMat, dst: imageMat, kernel: kernelForDilation)
        
        let edgesMat = Mat()
        Imgproc.Canny(image: imageMat, edges: edgesMat, threshold1: 100, threshold2: 200, apertureSize: 3)
//        return edgesMat.toUIImage()
        
        let contours: NSMutableArray = []
        let hierarchy = Mat()
        Imgproc.findContours(image: edgesMat, contours: contours, hierarchy: hierarchy, mode: .RETR_TREE, method: .CHAIN_APPROX_SIMPLE)
        
        let contourArray = contours.copy() as! [[Point2i]]
        
        // ↓検証用。輪郭の表示。
        /*
        let imageWithContours = Mat()
        original.copy(to: imageWithContours)
        let resizedContours = contourArray.map { contour in
            return contour.map { point in
                return Point2i(
                    x: Int32(Double(point.x) * (1.0 / resizeRatio)),
                    y: Int32(Double(point.y) * (1.0 / resizeRatio))
                )
            }
        }
        Imgproc.drawContours(
            image: imageWithContours,
//            contours: contours.copy() as! [[Point2i]],
            contours: resizedContours,
            contourIdx: -1,
            color: Scalar(0, 255, 0, 255),
            thickness: 24
        )
        return imageWithContours.toUIImage()
         */
        
        let sortedContourArray = contourArray.sorted(by: { contour1, contour2 in
            return contourArea(contour1) > contourArea(contour2)
        })
        let largestContours = sortedContourArray.prefix(10)
        
        // ↓検証用。エリアが最大の輪郭の表示。
        /*
        let imageWithContours = Mat()
        original.copy(to: imageWithContours)
        let largestContour = largestContours[0]
        let resizedLargestContour = largestContour.map { point in
            let resizeRatio2 = 1.0 / resizeRatio
            return Point2i(
                x: Int32(Double(point.x) * resizeRatio2),
                y: Int32(Double(point.y) * resizeRatio2)
            )
        }
        Imgproc.drawContours(
            image: imageWithContours,
            contours: [resizedLargestContour],
            contourIdx: -1,
            color: Scalar(0, 255, 0, 255),
            thickness: 24
        )
        return imageWithContours.toUIImage()
         */
        guard let receiptContour: [Point2i] = getReceiptContour(contours: Array(largestContours)) else {
            print("***** receiptContour is nil")
            return nil
        }
        
        // 検証
        /*
        let imageWithContours = Mat()
        original.copy(to: imageWithContours)
        let resizeRatio2 = 1.0 / resizeRatio
        let resizedContour = receiptContour.map { (point: Point2i) -> Point2i in
            let newX = Int32(Double(point.x) * resizeRatio2)
            let newY = Int32(Double(point.y) * resizeRatio2)
            return Point2i(x: newX, y: newY)
        }
        Imgproc.drawContours(
            image: imageWithContours,
            contours: [resizedContour],
            contourIdx: -1,
            color: Scalar(0, 255, 0, 255),
            thickness: 24
        )
        return imageWithContours.toUIImage()
         */
        
        let receiptRectInOriginal = contourToRect(contours: receiptContour, multiplier: 1.0 / resizeRatio)
        let transformedMat = warpPerspective(imageMat: original, rect: receiptRectInOriginal)
        
        let grayScaledMat = Mat()
        Imgproc.cvtColor(src: transformedMat, dst: grayScaledMat, code: .COLOR_BGR2GRAY)
        Imgproc.adaptiveThreshold(src: grayScaledMat, dst: grayScaledMat, maxValue: 255, adaptiveMethod: AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType: ThresholdTypes.THRESH_BINARY, blockSize: 21, C: 5)
        
        return grayScaledMat.toUIImage()
    }
    
    static func resize(mat: Mat, resizeRatio: Double) {
        let width = Int32(Double(mat.width()) * resizeRatio)
        let height = Int32(Double(mat.height()) * resizeRatio)
        Imgproc.resize(src: mat, dst: mat, dsize: Size2i(width: width, height: height))
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
    
    private static func contourArea(_ contour: [Point2i]) -> Double {
        let contourMat = intPointsToMat(points: contour)
        return Imgproc.contourArea(contour: contourMat, oriented: false)
    }
    
    private static func approximateContour(contour: [Point2i]) -> [Point2i] {
        let contourInPoint2f = point2isToPoint2fs(contour)
        let perimeterLength = Imgproc.arcLength(curve: contourInPoint2f, closed: true)
        let approxContour = NSMutableArray()
        Imgproc.approxPolyDP(curve: contourInPoint2f, approxCurve: approxContour, epsilon: 0.032 * perimeterLength, closed: true)
        return approxContour as! [Point2i]
    }
    
    private static func point2iToPoint2f(_ point2i: Point2i) -> Point2f {
        return Point2f(
            x: Float(point2i.x),
            y: Float(point2i.y)
        )
    }
    
    private static func point2isToPoint2fs(_ point2is: [Point2i]) -> [Point2f] {
        return point2is.map { point2iToPoint2f($0) }
    }
    
    private static func getReceiptContour(contours: [[Point2i]]) -> [Point2i]? {
        return contours.first { contour in
            let approx = approximateContour(contour: contour)
            print("***** approx.counnt: \(approx.count)")
            return approx.count == 4
        }
    }
    
    private static func contourToRect(contours: [Point2i], multiplier: Double = 1.0) -> [Point2i] {
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
            multiply(point: contours[minIndexXySumPerPointList], multiplier: multiplier),
            multiply(point: contours[minIndexYxDiffPerPointList], multiplier: multiplier),
            multiply(point: contours[maxIndexXySumPerPointList], multiplier: multiplier),
            multiply(point: contours[maxIndexYxDiffPerPointList], multiplier: multiplier),
        ]
    }
    
    private static func multiply(point: Point2i, multiplier: Double) -> Point2i {
        let newX = Int32(Double(point.x) * multiplier)
        let newY = Int32(Double(point.y) * multiplier)
        return Point2i(x: newX, y: newY)
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
}
