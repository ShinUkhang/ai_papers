// 2쪽 모아찍기: 왼쪽 = 원문 i쪽, 오른쪽 = 한국어 해설 i쪽 (가로 A4 한 장)
// 사용법: swift tools/side_by_side.swift <원문.pdf> <해설.pdf> <출력.pdf>
import Foundation
import PDFKit

let args = CommandLine.arguments
guard args.count == 4,
      let original = PDFDocument(url: URL(fileURLWithPath: args[1])),
      let notes = PDFDocument(url: URL(fileURLWithPath: args[2])) else {
    print("usage: side_by_side.swift original.pdf notes.pdf out.pdf"); exit(1)
}

var sheet = CGRect(x: 0, y: 0, width: 842, height: 595)  // 가로 A4 (pt)
let margin: CGFloat = 14
let half = (sheet.width - margin * 3) / 2
guard let ctx = CGContext(URL(fileURLWithPath: args[3]) as CFURL, mediaBox: &sheet, nil) else { exit(1) }

func draw(_ page: PDFPage?, in slot: CGRect) {
    guard let page = page, let cg = page.pageRef else { return }
    let box = page.bounds(for: .mediaBox)
    let scale = min(slot.width / box.width, slot.height / box.height)
    let w = box.width * scale, h = box.height * scale
    ctx.saveGState()
    ctx.translateBy(x: slot.minX + (slot.width - w) / 2, y: slot.minY + (slot.height - h) / 2)
    ctx.scaleBy(x: scale, y: scale)
    ctx.translateBy(x: -box.minX, y: -box.minY)
    ctx.drawPDFPage(cg)
    ctx.restoreGState()
    ctx.setStrokeColor(gray: 0.75, alpha: 1)
    ctx.stroke(CGRect(x: slot.minX + (slot.width - w) / 2, y: slot.minY + (slot.height - h) / 2, width: w, height: h))
}

let count = max(original.pageCount, notes.pageCount)
for i in 0..<count {
    ctx.beginPDFPage(nil)
    let left = CGRect(x: margin, y: margin, width: half, height: sheet.height - margin * 2)
    let right = CGRect(x: margin * 2 + half, y: margin, width: half, height: sheet.height - margin * 2)
    draw(i < original.pageCount ? original.page(at: i) : nil, in: left)
    draw(i < notes.pageCount ? notes.page(at: i) : nil, in: right)
    ctx.endPDFPage()
}
ctx.closePDF()
print("OK \(args[3]) (\(count) sheets)")
